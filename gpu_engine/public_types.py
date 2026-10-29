"""Public live-type overlays from Forecast and Color Change protocol facts.

The base team writer still owns species/Transform types. This consumer supplies
only an active-mon overlay, with an explicit dynamic write mask: an unresolved
forme falls back to the base writer instead of erasing its categories. Arrival
clears the overlay; faint and unrelated events preserve the source's value.
"""
from dataclasses import dataclass
import hashlib
import json
from typing import NamedTuple
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference
from pokezero.dex import ShowdownDex, SpeciesInfo
from pokezero.category_vocab import CategoryVocabulary
from .private_team import vocabulary_binding
from .public_moves import _static_category
from .public_mon_state import _int32
from .public_abilities import _bool

SCHEMA='pokezero.public_types.v1'
ARRIVE, CLEAR, TYPE, FORME=range(4)
COLUMNS=(reference.CATEGORY_TYPE_1,reference.CATEGORY_TYPE_2)


class TypeTables(NamedTuple):
    forme_categories: jax.Array
    forme_resolved: jax.Array
    type_category: jax.Array
    species_binding: jax.Array
    vocabulary_binding: jax.Array
    binding: jax.Array


@dataclass(frozen=True)
class TypeCatalog:
    species: object
    vocabulary: object
    dex: object
    device: TypeTables


def compile_types(species_catalog,dex,vocabulary):
    vb=vocabulary_binding(vocabulary)
    if not np.array_equal(np.asarray(vb),np.asarray(species_catalog.device.vocabulary_binding)):
        raise ValueError('live types and public species require the same frozen vocabulary')
    vocabulary=CategoryVocabulary(tuple(vocabulary.tokens),vocabulary.oov_buckets,
        MappingProxyType(dict(vocabulary.aliases)))
    object.__setattr__(vocabulary,'_index',MappingProxyType(dict(vocabulary._index)))
    # Freeze the pure type facts used by the lexical fallback as well as by
    # device lookups; mutating the caller's dex cannot change a bound producer.
    dex=ShowdownDex({},MappingProxyType({key:SpeciesInfo(info.id,info.name,tuple(info.types),MappingProxyType({}))
        for key,info in dex.species.items()}),{})
    bases=[]
    for name in species_catalog.names:
        info=reference._species_info_base_fallback(dex,name) if name else None
        types=info.types if info else ()
        bases.append([_static_category(vocabulary,'type:'+types[i]) if i<len(types) else 0 for i in (0,1)])
    if not np.array_equal(np.asarray(bases,np.int32),np.asarray(species_catalog.device.type_categories)):
        raise ValueError('live-type dex facts differ from the frozen public species catalog')
    categories=[];resolved=[]
    # Real dex forms; raw dex-absent forms use the same source fallback in the
    # lexical adapter and carry the resulting frozen type-category rows.
    labels={str(t) for info in dex.species.values() for t in info.types}
    labels.update(reference._FORMECHANGE_TYPE_FALLBACK.values())
    for name in species_catalog.names:
        pair=reference._resolve_live_type_slots('forme:'+name,dex) if name else None
        resolved.append(pair is not None)
        categories.append([_static_category(vocabulary,'type:'+t) if t else 0 for t in pair] if pair else [0,0])
    allowed=np.zeros(vocabulary.size,bool)
    for label in labels:allowed[_static_category(vocabulary,'type:'+label)]=True
    for label in vocabulary.tokens:
        if label.strip().lower().startswith('type:'):allowed[_static_category(vocabulary,label)]=True
    # Source accepts arbitrary public type strings; retain its exact frozen OOV
    # hash semantics rather than silently coercing an unfamiliar label to zero.
    allowed[1+len(vocabulary.tokens):]=True
    payload=dict(schema=SCHEMA,forms=categories,resolved=resolved,allowed=allowed.tolist(),
        raw_forme_types={key:list(info.types) for key,info in dex.species.items()},
        species=np.asarray(species_catalog.device.binding).tolist(),vocabulary=np.asarray(vb).tolist())
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    binding=jnp.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32))
    return TypeCatalog(species_catalog,vocabulary,dex,TypeTables(jnp.asarray(categories,jnp.int32),
        jnp.asarray(resolved,bool),jnp.asarray(allowed),species_catalog.device.binding,vb,binding))


class TypeState(NamedTuple):
    binding: jax.Array
    species_binding: jax.Array
    encounter: jax.Array
    overridden: jax.Array
    categories: jax.Array


def _binding(value):
    if getattr(value,'shape',None)!=(8,) or getattr(value,'dtype',None)!=jnp.dtype(jnp.uint32):
        raise ValueError('live-type namespace requires eight uint32 words')
    return value


def initialize(batch,tables,*,species_binding):
    if type(batch) is not int or batch<=0:raise ValueError('positive live-type batch required')
    _binding(species_binding)
    if not np.array_equal(np.asarray(species_binding),np.asarray(tables.species_binding)):
        raise ValueError('live types require their frozen public species namespace')
    return TypeState(jnp.broadcast_to(tables.binding,(batch,8)),jnp.broadcast_to(species_binding,(batch,8)),
        jnp.zeros((batch,2),jnp.int32),jnp.zeros((batch,2),bool),jnp.zeros((batch,2,2),jnp.int32))


def _shape(state,tables,mons):
    if state.encounter.ndim!=2 or state.encounter.shape[1]!=2 or not state.encounter.shape[0]:
        raise ValueError('live-type state requires B by two public sides')
    b=state.encounter.shape[0]
    for name,shape,dtype in [('binding',(b,8),jnp.uint32),('species_binding',(b,8),jnp.uint32),
        ('encounter',(b,2),jnp.int32),('overridden',(b,2),jnp.bool_),('categories',(b,2,2),jnp.int32)]:
        x=getattr(state,name)
        if x.shape!=shape or x.dtype!=dtype:raise ValueError('live-type state shape/dtype: '+name)
    for name,shape in [('active',(b,2)),('encounter',(b,2,6)),('species',(b,2,6))]:
        x=getattr(mons,name)
        if x.shape!=shape or x.dtype!=jnp.int32:raise ValueError('public encounter shape/dtype: '+name)
    if mons.species_binding.shape!=(b,8) or mons.species_binding.dtype!=jnp.uint32:
        raise ValueError('public species namespace shape/dtype')
    for x in (tables.binding,tables.species_binding,tables.vocabulary_binding):_binding(x)
    n=tables.forme_resolved.shape[0]
    if tables.forme_categories.shape!=(n,2) or tables.forme_categories.dtype!=jnp.int32 or tables.forme_resolved.dtype!=jnp.bool_:
        raise ValueError('frozen live-type forme table shape/dtype')
    if tables.type_category.ndim!=1 or not tables.type_category.shape[0] or tables.type_category.dtype!=jnp.bool_:
        raise ValueError('frozen live-type category membership shape/dtype')
    return b


def _coherent(state,tables,mons):
    b=state.encounter.shape[0];rows=jnp.arange(b)[:,None];sides=jnp.arange(2)[None,:]
    active=mons.active;slot=jnp.clip(active,0,5)
    current=jnp.where(active>=0,mons.encounter[rows,sides,slot],0)
    species=mons.species[rows,sides,slot]
    cats=state.categories;n=len(tables.type_category)
    allowed=tables.type_category[jnp.clip(cats,0,n-1)]
    valid_cats=(cats>=0)&(cats<n)&((cats==0)|allowed)
    ok=jnp.all(state.binding==tables.binding,axis=-1)&jnp.all(state.species_binding==tables.species_binding,axis=-1)
    ok &= jnp.all(mons.species_binding==tables.species_binding,axis=-1)&jnp.all((active>=-1)&(active<6),axis=-1)
    ok &= jnp.all((active<0)|((current>0)&(species>0)&(species<len(tables.forme_resolved))),axis=-1)
    ok &= jnp.all(valid_cats,axis=(-1,-2))&jnp.all(jnp.where(state.overridden,cats[...,0]>0,jnp.all(cats==0,axis=-1)),axis=-1)
    ok &= jnp.all((state.encounter>=0)&(~state.overridden|(state.encounter>0)),axis=-1)
    return current,species,ok


def apply(state,tables,mons,*,operation,side,encounter,first=0,second=0,producer_binding,valid=True):
    """Consume an exact current public encounter; refuse bad rows atomically."""
    b=_shape(state,tables,mons);_binding(producer_binding)
    op,seat,enc,x,y=(jnp.broadcast_to(_int32(v),(b,)) for v in (operation,side,encounter,first,second))
    valid=jnp.broadcast_to(_bool(valid),(b,))
    current,species,ok=_coherent(state,tables,mons)
    row=jnp.arange(b);target=jnp.clip(seat,0,1);address=(row,target);arrival=op==ARRIVE
    ok &= jnp.all(producer_binding==tables.binding)&(op>=ARRIVE)&(op<=FORME)&(seat>=0)&(seat<2)&(enc>0)&(current[address]==enc)
    same=state.encounter==current
    same=jnp.where(arrival[:,None]&(jnp.arange(2)==target[:,None]),True,same)
    ok &= jnp.all(same,axis=-1)
    n=len(tables.type_category)
    type_ok=(x>0)&(x<n)&(y>=0)&(y<n)&tables.type_category[jnp.clip(x,0,n-1)]&((y==0)|tables.type_category[jnp.clip(y,0,n-1)])
    ok &= jnp.where(op==TYPE,type_ok,jnp.where(op==FORME,(x>=0)&(x<len(tables.forme_resolved))&(y==0),(x==0)&(y==0)))
    form=jnp.clip(x,0,len(tables.forme_resolved)-1)
    form_valid=tables.forme_resolved[form]&(x!=species[address])
    overridden=(op==TYPE)|((op==FORME)&form_valid)
    pair=jnp.where((op==TYPE)[:,None],jnp.stack((x,y),axis=-1),tables.forme_categories[form])
    pair=jnp.where(overridden[:,None],pair,0)
    use=ok&valid
    updated=state._replace(encounter=state.encounter.at[address].set(jnp.where(use,enc,state.encounter[address])),
        overridden=state.overridden.at[address].set(jnp.where(use,overridden,state.overridden[address])),
        categories=state.categories.at[address].set(jnp.where(use[:,None],pair,state.categories[address])))
    return updated,ok|~valid


class TypePatch(NamedTuple):
    categorical: jax.Array       # [B,2,41], active public-side order
    write_mask: jax.Array        # [B,2,41], only resolved live overrides write
    admitted: jax.Array


def project(state,tables,mons,*,expected_vocabulary):
    _shape(state,tables,mons);_binding(expected_vocabulary)
    current,_,ok=_coherent(state,tables,mons)
    ok &= jnp.all(state.encounter==current,axis=-1)&jnp.all(expected_vocabulary==tables.vocabulary_binding)
    use=state.overridden&(current>0)&ok[:,None]
    cats=jnp.zeros((*state.encounter.shape,41),jnp.int32).at[...,jnp.asarray(COLUMNS)].set(jnp.where(use[...,None],state.categories,0))
    mask=jnp.zeros_like(cats,bool).at[...,jnp.asarray(COLUMNS)].set(use[...,None])
    return TypePatch(cats,mask,ok)


@dataclass(frozen=True)
class TypeProjection:
    raw_line: str
    facts: tuple[tuple[int,int,int,int],...]
    type_binding: tuple[int,...]
    full_line_consumers_complete: bool=False


def project_line(line,*,catalog):
    """Extract lexical type facts; canonical arrivals belong to the stream."""
    if not isinstance(line,str):raise ValueError('public type source line must be text')
    p=line.split('|');facts=[];kind=p[1] if len(p)>1 else ''
    slot=reference._slot_from_ident(p[2]) if len(p)>2 else None
    if slot in ('p1','p2'):
        side=int(slot[1])-1
        if kind=='-formechange' and len(p)>=4:
            key=reference._normalize_identifier(p[3].strip())
            ident=catalog.species.ids.get(key,0)
            pair=reference._resolve_live_type_slots('forme:'+p[3].strip(),catalog.dex)
            if ident or pair is None:facts.append((FORME,side,ident,0))
            else:
                a,c=pair;facts.append((TYPE,side,_static_category(catalog.vocabulary,'type:'+a),
                    _static_category(catalog.vocabulary,'type:'+c) if c else 0))
        elif kind=='-start' and len(p)>=5 and reference._normalize_identifier(p[3])=='typechange' and p[4].strip():
            pair=reference._resolve_live_type_slots('type:'+p[4].strip(),None)
            if pair is not None:
                a,c=pair;facts.append((TYPE,side,_static_category(catalog.vocabulary,'type:'+a),
                    _static_category(catalog.vocabulary,'type:'+c) if c else 0))
            else:facts.append((CLEAR,side,0,0))
        elif kind=='-end' and len(p)>=4 and reference._normalize_identifier(p[3])=='typechange':facts.append((CLEAR,side,0,0))
    return TypeProjection(line,tuple(facts),tuple(int(x) for x in np.asarray(catalog.device.binding)))
