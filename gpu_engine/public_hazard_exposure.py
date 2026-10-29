"""Champion field-token future Spikes exposure from public and owning facts.

This prices the source's living grounded bench population. It is not physical
grounding simulation, damage attribution, or a sampled opponent-team input.
"""
from dataclasses import dataclass
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference
from . import public_belief_features as features, public_field
from . import public_belief_snapshot as snapshot
from .private_team import vocabulary_binding
from .private_requests import parse_action_request
from .public_mon_state import _int32
from .public_abilities import _bool
from .public_moves import _static_category

SCHEMA='pokezero.public_hazard_exposure.v1'
COLUMNS=tuple(reference.v4_numeric_index(x) for x in (
    reference.NUMERIC_SELF_HAZARD_EXPECTED,reference.NUMERIC_OPP_HAZARD_EXPECTED))


class HazardTables(NamedTuple):
    exact_flying: jax.Array
    levitate: jax.Array
    species_binding: jax.Array
    vocabulary_binding: jax.Array
    feature_binding: jax.Array
    field_binding: jax.Array
    binding: jax.Array


@dataclass(frozen=True)
class HazardCatalog:
    species: object
    device: HazardTables


def compile_hazard(species_catalog,dex,vocabulary,*,feature_tables,field_tables,abilities):
    # Python declines an empty possible-ability filter after a pin, retaining
    # its prior list. A variant-only reduction cannot reconstruct that legacy
    # surface for blank ability entries; do not silently price it as grounded.
    if np.any(np.asarray(feature_tables.variant_valid)&(np.asarray(feature_tables.variant_ability)==0)):
        raise ValueError('hazard exposure requires nonempty semantic abilities in every valid source variant')
    vb=vocabulary_binding(vocabulary)
    for actual,expected in ((species_catalog.device.binding,feature_tables.species_binding),
            (species_catalog.device.vocabulary_binding,vb),(feature_tables.vocabulary_binding,vb),
            (field_tables.vocabulary_binding,vb),(feature_tables.ability_binding,abilities.device.binding)):
        if not np.array_equal(np.asarray(actual),np.asarray(expected)):
            raise ValueError('hazard exposure requires the exact species/features/ability/field/vocabulary namespaces')
    flying=[]
    for index,name in enumerate(species_catalog.names):
        info=dex.species_info(name) if name else None
        flying.append(info is not None and any(reference._normalize_identifier(x)=='flying' for x in info.types))
        # The base team catalog uses a fallback for formes; validate those
        # categories, but never use that fallback to grant Flying immunity.
        base=reference._species_info_base_fallback(dex,name) if name else None
        types=base.types if base else ()
        expected=[_static_category(vocabulary,'type:'+types[i]) if i<len(types) else 0 for i in range(2)]
        if list(np.asarray(species_catalog.device.type_categories[index]))!=expected:
            raise ValueError('hazard dex differs from the bound public species catalog')
    levitate=abilities.ids.get('levitate',0)
    payload=dict(schema=SCHEMA,flying=flying,levitate=levitate,columns=COLUMNS,
        species=np.asarray(species_catalog.device.binding).tolist(),vocabulary=np.asarray(vb).tolist(),
        features=np.asarray(feature_tables.binding).tolist(),field=np.asarray(field_tables.binding).tolist())
    h=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    return HazardCatalog(species_catalog,HazardTables(jnp.asarray(flying,bool),jnp.asarray(levitate,jnp.int32),
        species_catalog.device.binding,vb,feature_tables.binding,field_tables.binding,
        jnp.asarray(np.frombuffer(h,dtype='>u4').astype(np.uint32))))


class OwnHazardFacts(NamedTuple):
    owner_side: jax.Array
    species_binding: jax.Array
    vocabulary_binding: jax.Array
    species: jax.Array
    present: jax.Array
    active: jax.Array
    fainted: jax.Array
    levitate: jax.Array


def parse_own_hazard(request,*,owner_side,catalog,vocabulary):
    owner=parse_action_request(request,owner_side=owner_side).owner_side
    vb=vocabulary_binding(vocabulary)
    if not np.array_equal(np.asarray(vb),np.asarray(catalog.device.vocabulary_binding)):
        raise ValueError('own hazard facts require the frozen vocabulary')
    team=reference._self_team_from_request(request,f'p{owner_side+1}')
    species=np.zeros(6,np.int32);present=np.zeros(6,bool);active=np.zeros(6,bool);fainted=np.zeros(6,bool);levitate=np.zeros(6,bool)
    for i,mon in enumerate(team):
        if mon.showdown_slot!=f'p{owner_side+1}':raise ValueError('own hazard row belongs to another seat')
        species[i]=catalog.species.species_id(mon.species);present[i]=True;active[i]=mon.active
        fainted[i]=reference._condition_features(mon.condition).fainted
        levitate[i]=reference._normalize_identifier(mon.ability or '')=='levitate'
    return OwnHazardFacts(owner,catalog.device.species_binding,vb,
        *(jnp.asarray(x) for x in (species,present,active,fainted,levitate)))


class HazardPatch(NamedTuple):
    numeric: jax.Array
    numeric_columns: jax.Array
    admitted: jax.Array


def project(own,mons,context,beliefs,visible,source_tables,field,tables,feature_tables,*,observer_side,expected_binding):
    """Two field cells from owning rows and the already-materialized posterior.

    Read only the original public ability reveal, then a singleton semantic
    ability from that posterior. Trace/Transform running overrides are ignored.
    """
    if not isinstance(own,OwnHazardFacts):raise ValueError('owner-bound hazard request facts required')
    b=public_field._batch(field)
    observer=_int32(observer_side)
    if observer.shape not in ((),(b,)):raise ValueError('hazard observer must be scalar or exact batch')
    observer=jnp.broadcast_to(observer,(b,));seat=jnp.clip(observer,0,1);opp=1-seat;rows=jnp.arange(b)
    def packet(x,tail,dtype):
        if getattr(x,'dtype',None)!=jnp.dtype(dtype) or x.shape not in (tail,(b,)+tail):
            raise ValueError('own hazard packet shape/dtype')
        return jnp.broadcast_to(x,(b,)+tail)
    def binding(x):
        if getattr(x,'shape',None)!=(8,) or x.dtype!=jnp.uint32:raise ValueError('hazard namespace requires eight uint32 words')
    for x in tables[2:]:binding(x)
    binding(expected_binding)
    owner=packet(own.owner_side,(),jnp.int32);sb=packet(own.species_binding,(8,),jnp.uint32)
    vb=packet(own.vocabulary_binding,(8,),jnp.uint32);species=packet(own.species,(6,),jnp.int32)
    present=packet(own.present,(6,),jnp.bool_);active=packet(own.active,(6,),jnp.bool_)
    fainted=packet(own.fainted,(6,),jnp.bool_);levitate=packet(own.levitate,(6,),jnp.bool_)
    if tables.exact_flying.ndim!=1 or tables.exact_flying.dtype!=jnp.bool_ or tables.levitate.shape!=() or tables.levitate.dtype!=jnp.int32:
        raise ValueError('hazard static semantic table shape/dtype')
    ok=(observer>=0)&(observer<=1)&(owner==observer)&jnp.all(expected_binding==tables.binding)
    ok &= jnp.all(sb==tables.species_binding,axis=-1)&jnp.all(vb==tables.vocabulary_binding,axis=-1)
    ok &= jnp.all(field.binding==tables.field_binding,axis=-1)&jnp.all(feature_tables.binding==tables.feature_binding)
    ok &= jnp.all(feature_tables.species_binding==tables.species_binding)&jnp.all(feature_tables.vocabulary_binding==tables.vocabulary_binding)
    ok &= jnp.all((field.spikes>=0)&(field.spikes<=3),axis=-1)
    ok &= jnp.all(jnp.where(present,(species>0)&(species<tables.exact_flying.shape[0]),
        (species==0)&~active&~fainted&~levitate),axis=-1)
    # Never materialize at encode: pin registration can precede the source's
    # next snapshot refresh. Consume its explicitly carried device cache.
    summary,visible_ok=snapshot.read(visible,beliefs,source_tables)
    ok &= jnp.all(visible_ok[rows,seat],axis=-1)
    ok &= jnp.all(source_tables.binding==feature_tables.source_binding)
    validated=features.project_opponent(beliefs,summary,mons,context,feature_tables,
        observer_side=observer,expected_binding=tables.feature_binding)
    ok &= validated.patch.admitted
    own_flying=tables.exact_flying[jnp.clip(species,0,tables.exact_flying.shape[0]-1)]
    own_count=jnp.sum(present&~active&~fainted&~levitate&~own_flying,axis=-1)
    public_species=mons.species[rows,opp];seen=mons.encounter[rows,opp]>0
    ok &= jnp.all(~seen|((public_species>0)&(public_species<tables.exact_flying.shape[0])),axis=-1)
    if mons.fainted.shape!=(b,2,6) or mons.fainted.dtype!=jnp.bool_ or mons.active.shape!=(b,2) or mons.active.dtype!=jnp.int32:
        raise ValueError('hazard public active/faint state shape/dtype')
    ok &= jnp.all((mons.active>=-1)&(mons.active<6),axis=-1)
    source_species=beliefs.species[rows,seat];safe=jnp.clip(source_species,0,feature_tables.variant_ability.shape[0]-1)
    selected=features._bits(summary.effective[rows,seat],feature_tables.variant_ability.shape[1])
    variant_ability=feature_tables.variant_ability[safe]
    has_levitate=jnp.any(selected&(variant_ability==tables.levitate)&(tables.levitate>0),axis=-1)
    has_other=jnp.any(selected&(variant_ability>0)&(variant_ability!=tables.levitate),axis=-1)
    singleton=has_levitate&~has_other
    reveal=context.abilities.original[rows,opp]
    known_levitate=jnp.where(reveal>0,(reveal==tables.levitate)&(tables.levitate>0),singleton)
    flying=tables.exact_flying[jnp.clip(public_species,0,tables.exact_flying.shape[0]-1)]
    public_active=jnp.arange(6)[None,:]==mons.active[rows,opp,None]
    bench=seen&~public_active&~mons.fainted[rows,opp]&~flying&~known_levitate
    unseen=jnp.maximum(0,6-seen.sum(axis=-1));opp_count=unseen+bench.sum(axis=-1)
    counts=jnp.stack((own_count,opp_count),axis=-1)
    layers=jnp.stack((field.spikes[rows,seat],field.spikes[rows,opp]),axis=-1)
    fractions=jnp.array([0.,1/8,1/6,1/4],jnp.float32)[jnp.clip(layers,0,3)]
    values=jnp.minimum(1.,counts.astype(jnp.float32)*fractions/6)
    numeric=jnp.zeros((b,132),jnp.float32).at[:,jnp.array(COLUMNS)].set(jnp.where(ok[:,None],values,0))
    return HazardPatch(numeric,jnp.zeros(132,bool).at[jnp.array(COLUMNS)].set(True),ok)
