"""Source-conditioned v4 opponent expected stats on the device.

Static spread tables call the frozen Python generator core, including this
encoder's documented has-physical predicate. Runtime posterior selection,
bounds, maxima, fallback and normalization run on device. These are champion
input features, not assertions about a sampled opponent's true stats.
"""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference
from . import public_belief as belief
from .private_requests import parse_action_request
from .private_team import TeamFeaturePatch
from .public_abilities import _bool
from .public_belief_features import _bits
from .public_mon_state import _int32
from .public_moves import normalize_id

SCHEMA='pokezero.public_expected_stats.v1'
STATS=('hp','atk','def','spa','spd','spe')
CORE=('EXPECTED_HP','EXPECTED_HP_LOW','EXPECTED_HP_HIGH','EXPECTED_ATK','EXPECTED_ATK_LOW',
    'EXPECTED_ATK_HIGH','EXPECTED_DEF','EXPECTED_SPA','EXPECTED_SPD','EXPECTED_SPE')
NUMERIC_COLUMNS=tuple(reference.v4_numeric_index(getattr(reference,'NUMERIC_'+n)) for n in CORE)


class ExpectedTables(NamedTuple):
    base: jax.Array             # public dex species by six baseline base stats
    info_present: jax.Array
    exact_hp: jax.Array         # Transform uses exact dex lookup, not cosmetic fallback
    source_species: jax.Array   # frozen join from public dex identity to source identity
    spreads: jax.Array          # source species by level 0..100 by variant by stat, uint16
    spread_status: jax.Array    # 0 unavailable, 1 valid, 2 source illegal-spread exception
    pool: jax.Array
    variant_valid: jax.Array
    source_binding: jax.Array
    species_binding: jax.Array
    ability_binding: jax.Array
    vocabulary_binding: jax.Array
    binding: jax.Array


@dataclass(frozen=True)
class ExpectedCatalog:
    names: tuple[str,...]
    ids: Mapping[str,int]
    device: ExpectedTables


def compile_expected(source,dex,species_catalog):
    """Freeze source-ordered spreads for every level; never materialize a world.

    A source illegal-spread exception is retained separately from an unavailable
    spread. Runtime honors the first unusable selected candidate in source order:
    unavailable falls back, while an actually reached illegal candidate refuses.
    """
    if not np.array_equal(np.asarray(source.device.binding),np.asarray(species_catalog.device.source_binding)):
        raise ValueError('expected stats require the same frozen source')
    names=species_catalog.names
    base=np.asarray(species_catalog.device.base_stats)
    if np.any(~np.isfinite(base)) or np.any((base<0)|(base!=np.floor(base))) or np.any(base>65535):
        raise ValueError('expected base stats must be finite nonnegative uint16 facts')
    # Revalidate the dex against the already admitted public species catalog.
    present=[];exact_hp=[]
    for index,name in enumerate(names):
        info=reference._species_info_base_fallback(dex,name) if name else None
        row=[info.base_stats.get(s,0) if info else 0 for s in STATS]
        exact=dex.species_info(name) if name else None
        hp=exact.base_stats.get('hp',0) if exact else 0
        if not np.array_equal(base[index],row) or float(species_catalog.device.original_hp[index])!=hp:
            raise ValueError('expected stats dex differs from the frozen public species dex')
        present.append(info is not None);exact_hp.append(hp)
    maximum=source.device.variant_valid.shape[-1]
    spreads=np.zeros((len(source.species_names),101,maximum,6),np.uint16)
    status=np.zeros(spreads.shape[:-1],np.uint8)
    for sid,variants in enumerate(source.variants):
        info=reference._species_info_base_fallback(dex,source.species_names[sid]) if sid else None
        if info is None:continue
        for index,variant in enumerate(variants):
            raw=variant.to_summary();physical=reference._variant_has_physical_attack(dex,raw)
            for level in range(1,101):
                try:stats=reference._variant_spread_stats(info.base_stats,level,raw,physical)
                except ValueError:
                    status[sid,level,index]=2
                    continue
                if stats is None:continue
                values=[stats[s] for s in STATS]
                if any(type(v) is not int or not 0<=v<=65535 for v in values):
                    raise ValueError('source spread is outside the exact uint16 table')
                spreads[sid,level,index]=values;status[sid,level,index]=1
    source_ids=np.asarray([source.species_id(name) if name else 0 for name in names],np.int32)
    arrays=(base.astype(np.uint16),np.asarray(present,bool),np.asarray(exact_hp,np.uint16),source_ids,spreads,status,
        np.asarray(source.device.pool),np.asarray(source.device.variant_valid))
    payload=dict(schema=SCHEMA,names=names,source=np.asarray(source.device.binding).tolist(),
        species=np.asarray(species_catalog.device.binding).tolist(),
        abilities=np.asarray(species_catalog.device.ability_binding).tolist(),
        vocabulary=np.asarray(species_catalog.device.vocabulary_binding).tolist(),
        arrays=[dict(shape=a.shape,dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest()) for a in arrays])
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    binding=jnp.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32))
    return ExpectedCatalog(names,MappingProxyType(dict(species_catalog.ids)),ExpectedTables(
        *(jnp.asarray(a) for a in arrays),source.device.binding,species_catalog.device.binding,
        species_catalog.device.ability_binding,species_catalog.device.vocabulary_binding,binding))


class OwnTargets(NamedTuple):
    owner_side: jax.Array
    species_binding: jax.Array  # [8] or [B,8] after stacking owning packets
    vocabulary_binding: jax.Array
    species: jax.Array          # own request order, opaque public dex IDs
    stats: jax.Array            # own actual values, not opponent private facts
    known: jax.Array            # required copied stat keys existed in source request


def parse_own_targets(request,*,owner_side,catalog):
    """Decode only the owning request; source Transform lookup is last-by-species.

    Species IDs are separate from aliased/OOV vocabulary IDs. This packet is not
    an assertion that Transform occurred; the public ability ledger owns that.
    """
    parse_action_request(request,owner_side=owner_side)
    team=reference._self_team_from_request(request,f'p{owner_side+1}')
    ids=np.zeros(6,np.int32);stats=np.zeros((6,6),np.float32);known=np.zeros((6,6),bool)
    for index,mon in enumerate(team):
        name=normalize_id(mon.species or '')
        if name and name not in catalog.ids:raise ValueError('own Transform target outside frozen public species catalog')
        ids[index]=catalog.ids.get(name,0)
        for column,key in enumerate(STATS):
            value=(mon.stats or {}).get(key)
            if value is None:continue
            if (isinstance(value,bool) or not isinstance(value,(int,float)) or value<0
                    or value>float(np.finfo(np.float32).max) or not np.isfinite(float(value))):
                raise ValueError('own Transform stat must be finite and nonnegative')
            stats[index,column]=value;known[index,column]=True
    return OwnTargets(jnp.asarray(owner_side,jnp.int32),catalog.device.species_binding,
        catalog.device.vocabulary_binding,jnp.asarray(ids),jnp.asarray(stats),jnp.asarray(known))


def _require(array,shape,dtype):
    if getattr(array,'shape',None)!=shape or getattr(array,'dtype',None)!=jnp.dtype(dtype):
        raise ValueError('expected stat shape or dtype mismatch')


def _row(value,batch,*,boolean=False):
    arr=_bool(value) if boolean else _int32(value)
    if arr.shape not in ((),(batch,)):raise ValueError('expected stat row requires scalar or exact batch vector')
    return jnp.broadcast_to(arr,(batch,))


def project_opponent(states,summary,mons,abilities,tables,*,observer_side,expected_binding,
                     own_targets=None,exact_state=True):
    """Write ten columns from a coherent materialized public posterior.

    Unknown Transform targets emit a zero block as Python does. If supplied,
    owning target packets must match this observer and both namespaces. This
    component alone does not prove raw-event or whole-observation completeness.
    """
    if mons.encounter.ndim!=3 or mons.encounter.shape[1:]!=(2,6) or not mons.encounter.shape[0]:
        raise ValueError('expected stats require a nonempty public encounter batch')
    batch=mons.encounter.shape[0];shape=(batch,2,6)
    for a in (mons.encounter,mons.species,mons.source_species,mons.level,abilities.transform_species,
              states.species,summary.count,summary.reason):_require(a,shape,jnp.int32)
    for a in (states.hard,summary.effective):_require(a,(*shape,2),jnp.uint32)
    _require(abilities.transformed,shape,jnp.bool_)
    _require(summary.uncertainty,shape,jnp.float32)
    for a in (mons.species_binding,abilities.binding,abilities.species_binding):_require(a,(batch,8),jnp.uint32)
    _require(expected_binding,(8,),jnp.uint32)
    observer=_row(observer_side,batch);exact=_row(exact_state,batch,boolean=True)
    rows=jnp.arange(batch);obs=jnp.clip(observer,0,1);side=1-obs;addr=(rows,side)
    present=mons.encounter[addr]>0;species=mons.species[addr];levels=mons.level[addr]
    source=states.species[rows,obs];count=summary.count[rows,obs];effective=summary.effective[rows,obs]
    pub=jnp.clip(species,0,tables.base.shape[0]-1);sid=jnp.clip(source,0,tables.spreads.shape[0]-1)
    level=jnp.clip(levels,1,100);maximum=tables.spreads.shape[2]
    ok=(observer>=0)&(observer<=1)&jnp.all(expected_binding==tables.binding)
    ok &= jnp.all(mons.species_binding==tables.species_binding,axis=-1)&jnp.all(abilities.species_binding==tables.species_binding,axis=-1)
    ok &= jnp.all(abilities.binding==tables.ability_binding,axis=-1)
    ok &= jnp.all((source>=0)&(source<tables.spreads.shape[0])&(source==mons.source_species[addr]),axis=-1)
    ok &= jnp.all(~present|((species>0)&(species<tables.base.shape[0])&(levels>=1)&(levels<=100)),axis=-1)
    ok &= jnp.all(~present|(source==tables.source_species[pub]),axis=-1)
    pool=tables.pool[sid];selected=_bits(effective,maximum)
    ok &= ~jnp.any(((effective|states.hard[rows,obs])&~pool)!=0,axis=(1,2))
    ok &= ~jnp.any(selected&~tables.variant_valid[sid],axis=(1,2))
    ok &= jnp.all(selected.sum(axis=-1)==count,axis=-1)
    u=summary.uncertainty[rows,obs];reason=summary.reason[rows,obs]
    ok &= jnp.all(jnp.isfinite(u)&(u>=0)&(u<=1)&(reason>=belief.RESOLVED)&(reason<=belief.NO_SOURCE),axis=-1)
    # Source order matters when the spread core stops on an unavailable row.
    status=tables.spread_status[sid,level]
    first=jnp.min(jnp.where(selected&(status!=1),jnp.arange(maximum),maximum),axis=-1)
    first_status=jnp.take_along_axis(jnp.pad(status,[(0,0),(0,0),(0,1)],constant_values=1),first[...,None],axis=-1)[...,0]
    transformed=abilities.transformed[addr]
    donor=abilities.transform_species[addr]
    # A true Transform with no donor cannot be a coherent resolved public
    # Transform event. Refuse instead of changing the source's transformed flag.
    ok &= jnp.all(~(present&transformed)|((donor>0)&(donor<tables.base.shape[0])),axis=-1)
    ok &= ~jnp.any(present&~transformed&exact[:,None]&(first_status==2),axis=-1)
    use_spreads=(count>0)&(first==maximum)
    spread=tables.spreads[sid,level].astype(jnp.int32)
    low=jnp.min(jnp.where(selected[...,None],spread,65535),axis=-2)
    high=jnp.max(jnp.where(selected[...,None],spread,0),axis=-2)
    base=tables.base[pub].astype(jnp.int32)
    baseline=((2*base+31+85//4)*level[...,None])//100+5
    baseline=baseline.at[...,0].set(((2*base[...,0]+31+85//4)*level)//100+level+10)
    hp,atk=baseline[...,0],baseline[...,1]
    normal=jnp.stack((hp,jnp.where(use_spreads,low[...,0],hp),jnp.where(use_spreads,high[...,0],hp),
        atk,jnp.where(use_spreads,low[...,1],atk),jnp.where(use_spreads,high[...,1],atk),
        *(jnp.where(base[...,i]>0,jnp.where(use_spreads,high[...,i],baseline[...,i]),0) for i in range(2,6))),axis=-1)
    normal=normal.at[...,:6].set(jnp.where(((base[...,0]>0)&(base[...,1]>0))[...,None],normal[...,:6],0))
    normal=jnp.where(tables.info_present[pub][...,None],normal,0).astype(jnp.float32)
    copied=jnp.zeros((batch,6,10),jnp.float32)
    if own_targets is not None:
        leading=own_targets.species.shape[:-1]
        if leading not in ((),(batch,)):raise ValueError('own target packet requires scalar or exact batch shape')
        _require(own_targets.species,(*leading,6),jnp.int32)
        _require(own_targets.stats,(*leading,6,6),jnp.float32)
        _require(own_targets.known,(*leading,6,6),jnp.bool_)
        owner=_row(own_targets.owner_side,batch)
        for actual,expected in ((own_targets.species_binding,tables.species_binding),
                                (own_targets.vocabulary_binding,tables.vocabulary_binding)):
            if getattr(actual,'shape',None) not in ((8,),(batch,8)) or getattr(actual,'dtype',None)!=jnp.dtype(jnp.uint32):
                raise ValueError('own target namespace requires eight uint32 words per packet')
            ok &= jnp.all(jnp.broadcast_to(actual,(batch,8))==expected,axis=-1)
        ids=jnp.broadcast_to(own_targets.species,(batch,6))
        stats=jnp.broadcast_to(own_targets.stats,(batch,6,6));known=jnp.broadcast_to(own_targets.known,(batch,6,6))
        ok &= owner==observer
        ok &= jnp.all((ids>=0)&(ids<tables.base.shape[0]),axis=-1)&jnp.all(jnp.isfinite(stats)&(stats>=0),axis=(1,2))
        # Match the source's dictionary comprehension: later duplicate species
        # replaces an earlier request member, even when the later stats are absent.
        match=(donor[...,None]==ids[:,None,:])&(ids[:,None,:]>0)
        index=jnp.max(jnp.where(match,jnp.arange(6),-1),axis=-1)
        target=stats[rows[:,None],jnp.clip(index,0,5)]
        complete=(index>=0)&jnp.all(known[rows[:,None],jnp.clip(index,0,5),1:],axis=-1)
        hp_base=tables.exact_hp[pub].astype(jnp.int32)
        trans_hp=jnp.where(hp_base>0,((2*hp_base+31+85//4)*level)//100+level+10,0)
        copied=jnp.stack((trans_hp,trans_hp,trans_hp,target[...,1],target[...,1],target[...,1],
            *(target[...,i] for i in range(2,6))),axis=-1)
        copied=jnp.where(complete[...,None],copied,0)
    values=jnp.minimum(1.,jnp.where(transformed[...,None],copied,normal)/reference._ACTUAL_STAT_DIVISOR)
    values=jnp.where((present&exact[:,None]&ok[:,None])[...,None],values,0.)
    numeric=jnp.zeros((batch,6,132),jnp.float32).at[...,jnp.asarray(NUMERIC_COLUMNS)].set(values)
    return TeamFeaturePatch(numeric,jnp.zeros((batch,6,41),jnp.int32),
        jnp.zeros(132,bool).at[jnp.asarray(NUMERIC_COLUMNS)].set(True),jnp.zeros(41,bool),ok)
