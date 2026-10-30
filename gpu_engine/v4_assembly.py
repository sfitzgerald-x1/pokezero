"""Player-relative assembly of the champion's 23-token observation.

Inputs are named device writers, not sampled engine worlds. Coverage travels
with each cell: placing a partial writer in a tensor does not qualify missing
columns or the recorded-decision/model gates. The cost placeholder stays a
separate API until those gates have authoritative evidence.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import private_actions, private_requests, private_team
from . import public_field, public_team
from . import public_moves,public_abilities
from pokezero import showdown as reference

TOKEN_TYPES=(0,)+(1,)*6+(2,)*6+(3,)*9+(5,)


class Assembly(NamedTuple):
    numeric_features: jax.Array
    categorical_ids: jax.Array
    token_type_ids: jax.Array
    attention_mask: jax.Array
    legal_action_mask: jax.Array
    switch_targets: jax.Array
    numeric_written: jax.Array
    categorical_written: jax.Array
    admitted: jax.Array
    category_count: jax.Array


def _array(value,shape,dtype,name):
    if getattr(value,'shape',None)!=shape or getattr(value,'dtype',None)!=jnp.dtype(dtype):
        raise ValueError(name+' shape/dtype mismatch')
    return value


def _mask(value,shape):
    if getattr(value,'dtype',None)!=jnp.bool_ or value.shape not in ((shape[-1],),shape):
        raise ValueError('writer coverage must be bool column vector or exact cell shape')
    return jnp.broadcast_to(value,shape)


def assemble_sections(*,field,own,opponent,actions,tendency,request,observer_side,
        opponent_present,category_count):
    """Place fixed named sections; legality and attention remain distinct.

    Own rows retain request party order. Opponent rows retain public encounter
    order. Illegal/absent action tokens still have attention, as in Python;
    switching targets are the owner's canonical party indices, not public IDs.
    A true ``admitted`` means packet composition, not complete encoder parity.
    """
    if type(category_count) is not int or category_count<=0:raise ValueError('frozen vocabulary size required')
    if request.owner_side.ndim!=1:raise ValueError('assembly requires batched owning requests')
    b=request.owner_side.shape[0]
    if not b:raise ValueError('nonempty observation batch required')
    observer=_array(observer_side,(b,),jnp.int32,'observer')
    _array(opponent_present,(b,6),jnp.bool_,'opponent present')
    numeric=jnp.zeros((b,23,132),jnp.float32);categorical=jnp.zeros((b,23,41),jnp.int32)
    nw=jnp.zeros_like(numeric,dtype=bool);cw=jnp.zeros_like(categorical,dtype=bool)
    admitted=(observer>=0)&(observer<2)&(request.owner_side==observer)
    for start,count,patch in ((0,1,field),(1,6,own),(7,6,opponent)):
        shape=(b,132) if count==1 else (b,count,132)
        cshape=shape[:-1]+(41,)
        nums=_array(patch.numeric,shape,jnp.float32,'section numeric')
        cats=_array(patch.categorical,cshape,jnp.int32,'section categorical')
        nm=_mask(patch.numeric_columns,shape);cm=_mask(patch.categorical_columns,cshape)
        use=(slice(None),slice(start,start+count))
        numeric=numeric.at[use].set(nums.reshape(b,count,132))
        categorical=categorical.at[use].set(cats.reshape(b,count,41))
        nw=nw.at[use].set(nm.reshape(b,count,132));cw=cw.at[use].set(cm.reshape(b,count,41))
        admitted &= _array(patch.admitted,(b,),jnp.bool_,'section admission')
        axes=tuple(range(1,len(shape)))
        admitted &= jnp.all(nm|(nums==0),axis=axes)&jnp.all(cm|(cats==0),axis=axes)
    _array(actions.numeric,(b,9,132),jnp.float32,'actions numeric')
    _array(actions.categorical,(b,9,41),jnp.int32,'actions categorical')
    _array(actions.legal,(b,9),jnp.bool_,'action legal')
    numeric=numeric.at[:,13:22].set(actions.numeric)
    categorical=categorical.at[:,13:22].set(actions.categorical)
    # private_actions is a complete nine-token writer, including source zero
    # defaults outside its written fields. This does not cover mon/field cells.
    nw=nw.at[:,13:22].set(True);cw=cw.at[:,13:22].set(True)
    admitted &= _array(actions.admitted,(b,),jnp.bool_,'actions admission')
    _array(tendency.stats_numeric,(b,132),jnp.float32,'stats numeric')
    _array(tendency.stats_role,(b,),jnp.int32,'stats role')
    _array(tendency.stats_attention,(b,),jnp.bool_,'stats attention')
    numeric=numeric.at[:,22].set(tendency.stats_numeric)
    categorical=categorical.at[:,22,2].set(tendency.stats_role)
    stats_mask=_mask(tendency.stats_columns,(b,132))
    nw=nw.at[:,22].set(stats_mask)
    cw=cw.at[:,22,2].set(True)
    admitted &= _array(tendency.admitted,(b,),jnp.bool_,'tendency admission')
    admitted &= jnp.all(stats_mask|(tendency.stats_numeric==0),axis=-1)
    tendency_numeric=_array(tendency.numeric,(b,6,132),jnp.float32,'mon tendency numeric')
    tendency_mask=_mask(tendency.numeric_columns,(b,6,132))
    admitted &= jnp.all(tendency_mask|(tendency_numeric==0),axis=(1,2))
    numeric=numeric.at[:,7:13].set(jnp.where(tendency_mask,tendency_numeric,numeric[:,7:13]))
    nw=nw.at[:,7:13].set(nw[:,7:13]|tendency_mask)
    inputs=private_requests.action_inputs(request,observer_side=observer)
    admitted &= jnp.all(inputs.legal==actions.legal,axis=-1)
    admitted &= jnp.all(jnp.isfinite(numeric),axis=(1,2))
    admitted &= jnp.all((categorical>=0)&(categorical<category_count),axis=(1,2))
    # NONE/WAIT requests have a valid observation layout, but cannot be used
    # as a policy decision. Keep this admission separate from attention.
    admitted &= inputs.ready
    attention=jnp.ones((b,23),bool)
    attention=attention.at[:,1:7].set(jnp.arange(6)[None,:]<request.team_size[:,None])
    attention=attention.at[:,7:13].set(opponent_present).at[:,22].set(tendency.stats_attention)
    return Assembly(numeric,categorical,jnp.broadcast_to(jnp.asarray(TOKEN_TYPES,jnp.int32),(b,23)),
        attention,actions.legal&admitted[:,None],jnp.where(admitted[:,None],inputs.switch_targets,-1),nw,cw,admitted,
        jnp.asarray(category_count,jnp.int32))


def overlay(result,patch,*,section):
    """Apply a named role writer with its own per-cell coverage and admission."""
    sections={'field':(0,1),'own':(1,6),'opponent':(7,6)}
    if section not in sections:raise ValueError('named field/own/opponent section required')
    start,count=sections[section];b=result.numeric_features.shape[0]
    shape=(b,132) if count==1 else (b,count,132);cshape=shape[:-1]+(41,)
    nums=_array(patch.numeric,shape,jnp.float32,'overlay numeric')
    cats=_array(patch.categorical,cshape,jnp.int32,'overlay categorical')
    nm=_mask(patch.numeric_columns,shape).reshape(b,count,132)
    cm=_mask(patch.categorical_columns,cshape).reshape(b,count,41)
    use=(slice(None),slice(start,start+count))
    numeric=result.numeric_features.at[use].set(jnp.where(nm,nums.reshape(b,count,132),result.numeric_features[use]))
    categorical=result.categorical_ids.at[use].set(jnp.where(cm,cats.reshape(b,count,41),result.categorical_ids[use]))
    admitted=result.admitted&_array(patch.admitted,(b,),jnp.bool_,'overlay admission')
    admitted &= jnp.all(~nm|jnp.isfinite(nums.reshape(b,count,132)),axis=(1,2))
    admitted &= jnp.all(~cm|((cats.reshape(b,count,41)>=0)&(cats.reshape(b,count,41)<result.category_count)),axis=(1,2))
    return result._replace(numeric_features=numeric,categorical_ids=categorical,
        numeric_written=result.numeric_written.at[use].set(result.numeric_written[use]|nm),
        categorical_written=result.categorical_written.at[use].set(result.categorical_written[use]|cm),
        admitted=admitted,legal_action_mask=result.legal_action_mask&admitted[:,None],
        switch_targets=jnp.where(admitted[:,None],result.switch_targets,-1))


def remap_public_own(patch,mons,own_status,*,observer_side):
    """Gather a full public per-mon writer into the private request's order.

    Join exact public species IDs, never checkpoint categories (aliases/OOV
    buckets may collide). Missing public encounters retain zeros; duplicate
    source identities or a public own mon absent from the request refuse.
    """
    b=mons.species.shape[0];observer=_array(observer_side,(b,),jnp.int32,'observer')
    owner=_array(own_status.owner_side,(b,),jnp.int32,'owner')
    species=_array(own_status.species,(b,6),jnp.int32,'own species')
    present=_array(own_status.present,(b,6),jnp.bool_,'own present')
    rows=jnp.arange(b);seat=jnp.clip(observer,0,1)
    public_species=mons.species[rows,seat];encountered=mons.encounter[rows,seat]>0
    match=(species[:,:,None]==public_species[:,None,:])&present[:,:,None]&encountered[:,None,:]
    ok=(owner==observer)&(observer>=0)&(observer<2)&patch.admitted
    binding=own_status.species_binding
    if binding.shape not in ((8,),(b,8)) or binding.dtype!=jnp.uint32:raise ValueError('own species namespace required')
    ok &= jnp.all(binding==mons.species_binding,axis=-1)
    ok &= jnp.all(match.sum(axis=-1)<=1,axis=-1)&jnp.all(~encountered|(match.sum(axis=1)==1),axis=-1)
    index=jnp.argmax(match,axis=-1);visible=jnp.any(match,axis=-1)
    nums=_array(patch.numeric,(b,2,6,132),jnp.float32,'public overlay numeric')
    cats=_array(patch.categorical,(b,2,6,41),jnp.int32,'public overlay categorical')
    nm=_mask(patch.numeric_columns,nums.shape)[rows[:,None],seat[:,None],index]
    cm=_mask(patch.categorical_columns,cats.shape)[rows[:,None],seat[:,None],index]
    numeric=nums[rows[:,None],seat[:,None],index];categorical=cats[rows[:,None],seat[:,None],index]
    return private_team.TeamFeaturePatch(jnp.where(visible[...,None],numeric,0),
        jnp.where(visible[...,None],categorical,0),nm&visible[...,None],cm&visible[...,None],ok)


def project_base_sections(*,facts,action_tables,mons,context,team_tables,field_tables,
        tendency,observer_side,vocabulary_binding,category_count,exact_state=True):
    """Run the existing source writers and assemble their real base sections."""
    own=private_team.project_team(facts.team,observer_side=observer_side,expected_vocabulary_binding=vocabulary_binding)
    retained_status = getattr(context, 'sleep', None)
    opponent=public_team.project_opponent(mons,context.abilities,team_tables,
        observer_side=observer_side,expected_binding=team_tables.binding,
        belief_status=retained_status.status if retained_status is not None else None)
    actions=private_actions.project_actions(facts,action_tables,observer_side=observer_side)
    field=public_field.project(context.field,field_tables,facts.request,observer_side=observer_side,
        expected_vocabulary=vocabulary_binding,exact_state=exact_state)
    present=mons.encounter[jnp.arange(mons.encounter.shape[0]),1-jnp.clip(observer_side,0,1)]>0
    result=assemble_sections(field=field,own=own,opponent=opponent,actions=actions,tendency=tendency,
        request=facts.request,observer_side=observer_side,opponent_present=present,category_count=category_count)
    bound=jnp.asarray(True)
    for catalog in (team_tables,action_tables,field_tables):
        binding=_array(catalog.vocabulary_binding,(8,),jnp.uint32,'writer vocabulary')
        bound &= jnp.all(binding==vocabulary_binding)
    admitted=result.admitted&bound
    return result._replace(admitted=admitted,legal_action_mask=result.legal_action_mask&admitted[:,None],
        switch_targets=jnp.where(admitted[:,None],result.switch_targets,-1))


def overlay_public_team(result,patch,mons,own_status,*,observer_side):
    """Compose both views of one public per-encounter writer explicitly."""
    own=remap_public_own(patch,mons,own_status,observer_side=observer_side)
    result=overlay(result,own,section='own')
    b=mons.encounter.shape[0];rows=jnp.arange(b);opponent=1-jnp.clip(observer_side,0,1)
    numeric=_array(patch.numeric,(b,2,6,132),jnp.float32,'public mon numeric')
    categorical=_array(patch.categorical,(b,2,6,41),jnp.int32,'public mon categorical')
    opp=private_team.TeamFeaturePatch(numeric[rows,opponent],categorical[rows,opponent],
        _mask(patch.numeric_columns,numeric.shape)[rows,opponent],
        _mask(patch.categorical_columns,categorical.shape)[rows,opponent],patch.admitted)
    return overlay(result,opp,section='opponent')


def overlay_public_active(result,patch,mons,own_status,*,observer_side):
    """Put side-keyed active values only on the matching active mon tokens."""
    b=mons.encounter.shape[0]
    numeric=_array(patch.numeric,(b,2,132),jnp.float32,'public active numeric')
    categorical=_array(patch.categorical,(b,2,41),jnp.int32,'public active categorical')
    active=(jnp.arange(6)[None,None,:]==mons.active[:,:,None])&(mons.encounter>0)
    nums=jnp.where(active[...,None],numeric[:,:,None,:],0)
    cats=jnp.where(active[...,None],categorical[:,:,None,:],0)
    nm=active[...,None]&_mask(patch.numeric_columns,numeric.shape)[:,:,None,:]
    cm=active[...,None]&_mask(patch.categorical_columns,categorical.shape)[:,:,None,:]
    full=private_team.TeamFeaturePatch(nums,cats,nm,cm,patch.admitted)
    return overlay_public_team(result,full,mons,own_status,observer_side=observer_side)


def project_active_categories(context,tables,mons,*,feature_pack_last_move=True):
    """Checkpoint categories39/40 use current side state, never mon set IDs."""
    b=mons.encounter.shape[0]
    last,move_ok=public_moves.last_move_categories(context.moves,tables.moves,mons.arrived_by_baton)
    traced,ability_ok=public_abilities.traced_categories(context.abilities,tables.abilities)
    enabled=jnp.broadcast_to(jnp.asarray(feature_pack_last_move,bool),(b,))
    cats=jnp.zeros((b,2,41),jnp.int32)
    cats=cats.at[:,:,reference.CATEGORY_LAST_USED_MOVE].set(jnp.where(enabled[:,None],last,0))
    cats=cats.at[:,:,reference.CATEGORY_TRACED_ABILITY].set(traced)
    columns=jnp.zeros(41,bool).at[jnp.array([reference.CATEGORY_LAST_USED_MOVE,reference.CATEGORY_TRACED_ABILITY])].set(True)
    return private_team.TeamFeaturePatch(jnp.zeros((b,2,132),jnp.float32),cats,jnp.zeros(132,bool),columns,move_ok&ability_ok)


def project_own_boosts(mons):
    """Source active boost columns for request-order own tokens after remap."""
    b=mons.encounter.shape[0]
    columns=jnp.array([reference.v4_numeric_index(getattr(reference,'NUMERIC_BOOST_'+name))
        for name in ('ATK','DEF','SPA','SPD','SPE')])
    numeric=jnp.zeros((b,2,132),jnp.float32).at[:,:,columns].set(jnp.clip(mons.boosts[:,:,:5]/6.,-1.,1.))
    valid=jnp.all((mons.boosts>=-6)&(mons.boosts<=6),axis=(1,2))
    return private_team.TeamFeaturePatch(numeric,jnp.zeros((b,2,41),jnp.int32),
        jnp.zeros(132,bool).at[columns].set(True),jnp.zeros(41,bool),valid)
