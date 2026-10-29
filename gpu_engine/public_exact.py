"""Rest wake certainty and known living bench trappers from public beliefs.

The live semantic ability set is reduced before display-bucket truncation.
Explicit exclusions are a separate input: inconsistent-posterior fallback must
not resurrect a ruled-out ability. Running Trace/Transform overrides are not
original-set evidence. This component does not produce exclusions itself.
"""
from dataclasses import dataclass
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference
from . import public_belief_features as features, public_sleep, public_mon_state as mon
from .private_team import TeamFeaturePatch, vocabulary_binding
from .private_requests import parse_action_request
from .public_abilities import _bool

SCHEMA='pokezero.public_exact.v1'
COLUMNS=tuple(reference.v4_numeric_index(x) for x in
    (reference.NUMERIC_WAKE_KNOWN,reference.NUMERIC_TRAPPER_ALIVE))
TRAPPERS=frozenset(('arenatrap','shadowtag','magnetpull'))


class Tables(NamedTuple):
    trapper: jax.Array
    early_bird: jax.Array
    feature_binding: jax.Array
    sleep_binding: jax.Array
    species_binding: jax.Array
    ability_binding: jax.Array
    vocabulary_binding: jax.Array
    binding: jax.Array


@dataclass(frozen=True)
class Catalog:
    abilities: object
    device: Tables


def compile_exact(abilities,feature_tables,sleep_tables,vocabulary):
    vb=vocabulary_binding(vocabulary)
    for actual,expected in ((abilities.device.binding,feature_tables.ability_binding),
            (feature_tables.species_binding,sleep_tables.species_binding),
            (feature_tables.vocabulary_binding,vb),(sleep_tables.vocabulary_binding,vb)):
        if not np.array_equal(np.asarray(actual),np.asarray(expected)):
            raise ValueError('exact writer requires one ability/species/sleep/features/vocabulary namespace')
    trapper=[name in TRAPPERS for name in abilities.names]
    early=abilities.ids.get('earlybird',0)
    payload=dict(schema=SCHEMA,columns=COLUMNS,trapper=trapper,early_bird=early,
        features=np.asarray(feature_tables.binding).tolist(),sleep=np.asarray(sleep_tables.binding).tolist(),
        species=np.asarray(feature_tables.species_binding).tolist(),abilities=np.asarray(abilities.device.binding).tolist(),
        vocabulary=np.asarray(vb).tolist())
    h=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    return Catalog(abilities,Tables(jnp.asarray(trapper,bool),jnp.asarray(early,jnp.int32),
        feature_tables.binding,sleep_tables.binding,feature_tables.species_binding,abilities.device.binding,vb,
        jnp.asarray(np.frombuffer(h,dtype='>u4').astype(np.uint32))))


class OwnAbilities(NamedTuple):
    owner_side: jax.Array
    ability_binding: jax.Array
    vocabulary_binding: jax.Array
    present: jax.Array
    ability: jax.Array


def parse_own_abilities(request,*,owner_side,catalog,vocabulary):
    owner=parse_action_request(request,owner_side=owner_side).owner_side
    vb=vocabulary_binding(vocabulary)
    if not np.array_equal(np.asarray(vb),np.asarray(catalog.device.vocabulary_binding)):
        raise ValueError('own ability facts require the frozen vocabulary')
    team=reference._self_team_from_request(request,f'p{owner_side+1}')
    present=np.zeros(6,bool);ability=np.zeros(6,np.int32)
    for i,pokemon in enumerate(team):
        if pokemon.showdown_slot!=f'p{owner_side+1}':raise ValueError('own ability row belongs to another seat')
        present[i]=True
        ability[i]=catalog.abilities.ability_id(pokemon.ability) if pokemon.ability else 0
    return OwnAbilities(owner,catalog.device.ability_binding,vb,jnp.asarray(present),jnp.asarray(ability))


class Patch(NamedTuple):
    own: TeamFeaturePatch
    opponent: TeamFeaturePatch


def project(own_abilities,own_team,own_status,mons,context,beliefs,summary,feature_tables,sleep_tables,tables,*,
            observer_side,expected_binding,ruled_out_abilities,exact_state=True):
    """Exclusions use absolute public side/encounter and bound semantic IDs.

    A caller without an exclusion producer must explicitly supply zeros and
    retain that producer's existing source/stream refusal. Never infer negative
    evidence from the posterior or from two display buckets.
    """
    b=public_sleep._shape(context.sleep,sleep_tables,mons)
    observer=mon._int32(observer_side);exact=_bool(exact_state)
    if observer.shape not in ((),(b,)) or exact.shape not in ((),(b,)):
        raise ValueError('exact observer/mask must be scalar or exact batch')
    observer=jnp.broadcast_to(observer,(b,));exact=jnp.broadcast_to(exact,(b,))
    rows=jnp.arange(b);seat=jnp.clip(observer,0,1);opponent=1-seat
    for x in tables[2:]:public_sleep._binding(x)
    public_sleep._binding(expected_binding)
    count=tables.trapper.shape[0]
    if tables.trapper.shape!=(count,) or tables.trapper.dtype!=jnp.bool_ or count!=feature_tables.ability_category.shape[0]:
        raise ValueError('exact semantic ability table shape/dtype')
    if tables.early_bird.shape!=() or tables.early_bird.dtype!=jnp.int32:
        raise ValueError('exact Early Bird ID must be scalar int32')
    def packet(x,shape,dtype):
        if getattr(x,'shape',None)!=shape or getattr(x,'dtype',None)!=jnp.dtype(dtype):
            raise ValueError('exact request/exclusion shape/dtype')
        return x
    if not isinstance(own_abilities,OwnAbilities):raise ValueError('owner-bound raw ability facts required')
    owner=packet(own_abilities.owner_side,(b,),jnp.int32)
    ab=packet(own_abilities.ability_binding,(b,8),jnp.uint32)
    vb=packet(own_abilities.vocabulary_binding,(b,8),jnp.uint32)
    present=packet(own_abilities.present,(b,6),jnp.bool_)
    ability=packet(own_abilities.ability,(b,6),jnp.int32)
    for x in (own_team.present,own_team.active,own_team.fainted):packet(x,(b,6),jnp.bool_)
    packet(own_team.owner_side,(b,),jnp.int32);packet(own_team.vocabulary_binding,(b,8),jnp.uint32)
    excluded=packet(ruled_out_abilities,(b,2,6,count),jnp.bool_)
    sleep=public_sleep.project(context.sleep,sleep_tables,mons,observer_side=observer,
        expected_vocabulary=tables.vocabulary_binding,own_status=own_status,exact_state=exact)
    belief=features.project_opponent(beliefs,summary,mons,context,feature_tables,observer_side=observer,
        expected_binding=tables.feature_binding)
    ok=sleep.admitted&belief.patch.admitted&jnp.all(expected_binding==tables.binding)
    ok &= jnp.all(feature_tables.binding==tables.feature_binding)&jnp.all(sleep_tables.binding==tables.sleep_binding)
    ok &= jnp.all(feature_tables.species_binding==tables.species_binding)&jnp.all(feature_tables.ability_binding==tables.ability_binding)
    ok &= (owner==observer)&(own_team.owner_side==observer)&jnp.all(ab==tables.ability_binding,axis=-1)
    ok &= jnp.all(vb==tables.vocabulary_binding,axis=-1)&jnp.all(own_team.vocabulary_binding==tables.vocabulary_binding,axis=-1)
    ok &= jnp.all(present==own_team.present,axis=-1)&jnp.all(present==own_status.present,axis=-1)
    ok &= jnp.all(own_team.active==own_status.active,axis=-1)
    ok &= jnp.all((ability>=0)&(ability<count)&(present|(ability==0)),axis=-1)
    ok &= ~jnp.any(excluded[...,0],axis=(-1,-2))
    ok &= ~jnp.any(excluded&~(mons.encounter>0)[...,None],axis=(-1,-2,-3))
    ok &= (tables.early_bird>=0)&(tables.early_bird<count)
    # Reproduce possible_abilities minus the independent ruled-out ledger.
    source_species=beliefs.species[rows,seat]
    safe=jnp.clip(source_species,0,feature_tables.variant_ability.shape[0]-1)
    selected=features._bits(summary.effective[rows,seat],feature_tables.variant_ability.shape[1])
    ids=feature_tables.variant_ability[safe]
    first=features._first_occurrences(ids,selected,count)
    live=(first<ids.shape[-1])&~excluded[rows,opponent]
    live_count=live.sum(axis=-1)
    revealed=context.abilities.original[rows,opponent]
    certain=jnp.where(revealed>0,revealed,jnp.where(live_count==1,jnp.argmax(live,axis=-1),0))
    no_early=(tables.early_bird==0)|~live[...,jnp.clip(tables.early_bird,0,count-1)]
    wake=(revealed>0)|((live_count>0)&no_early)
    sleeping=sleep.numeric[...,public_sleep.COLUMNS[1]]>0
    opp_wake=sleeping[rows,opponent]&wake
    opp_active=jnp.arange(6)[None,:]==mons.active[rows,opponent,None]
    opp_trap=(mons.encounter[rows,opponent]>0)&~opp_active&~mons.fainted[rows,opponent]&tables.trapper[jnp.clip(certain,0,count-1)]
    # Sleep history joins by exact public species; request order and active
    # flags remain private and may differ from the public pointer.
    match=(own_status.species[:,:,None]==mons.species[rows,seat,None,:])&(mons.encounter[rows,seat,None,:]>0)
    own_wake=jnp.any(match&sleeping[rows,seat,None,:],axis=-1)&present
    own_trap=present&~own_team.active&~own_team.fainted&tables.trapper[jnp.clip(ability,0,count-1)]
    mask=jnp.zeros(132,bool).at[jnp.array(COLUMNS)].set(True)
    def patch(a,c):
        values=jnp.stack((a,c),axis=-1).astype(jnp.float32)
        numeric=jnp.zeros((b,6,132),jnp.float32).at[:,:,jnp.array(COLUMNS)].set(jnp.where((ok&exact)[:,None,None],values,0))
        return TeamFeaturePatch(numeric,jnp.zeros((b,6,41),jnp.int32),mask,jnp.zeros(41,bool),ok)
    return Patch(patch(own_wake,own_trap),patch(opp_wake,opp_trap))
