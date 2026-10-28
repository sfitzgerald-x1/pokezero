"""Own request facts and device construction of all nine v4 action tokens.

Static move mechanics come from the source dex, not engine IDs. Hidden Power's
request identity stays separate from its typed mechanics lookup. This writes the
action section only; production observations still require all other sections.
"""
from dataclasses import dataclass, replace
import hashlib
import json
from types import MappingProxyType
from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference
from pokezero.dex import normalize_id, resolve_move_base_power, resolve_move_effect, ShowdownDex
from . import private_requests, private_team
from .public_moves import _static_category
from .public_mon_state import _int32

SCHEMA = 'pokezero.private_actions.v1'
LOW_HP, HIGH_HP = 1, 2
# Every genuine Gen3 HP stat fits the source writer's stated maximum. The
# finite table reproduces Python float64 division/floor at discontinuities;
# float32 division alone can choose a different move-power bucket.
DEFAULT_HP_LIMIT = int(reference._ACTUAL_STAT_DIVISOR)
SWITCH_CORE = ('HP_FRACTION','ACTIVE','LEGAL','PRESENT','LEVEL',
               *('BASE_'+n.upper() for n in private_team.STATS),
               *('ACTUAL_'+n.upper() for n in private_team.STATS))
SWITCH_COLUMNS = tuple(reference.v4_numeric_index(getattr(reference,'NUMERIC_'+n)) for n in SWITCH_CORE)
MOVE_CORE = ('BASE_POWER','PRIORITY','ACCURACY','EFFECT_CHANCE','SELF_HP_COST')
MOVE_COLUMNS = tuple(reference.v4_numeric_index(getattr(reference,'NUMERIC_'+n)) for n in MOVE_CORE)
MECHANIC_CATEGORIES = (reference.CATEGORY_TYPE_1,reference.CATEGORY_MOVE_CATEGORY,
                       reference.CATEGORY_MOVE_PRIORITY,reference.CATEGORY_MOVE_EFFECT)


class ActionTables(NamedTuple):
    numeric: jax.Array             # [move,ghost?,5] source-resolved static fields
    categories: jax.Array          # [move,ghost?,4] vocabulary rows
    hp_kind: jax.Array
    hp_power: jax.Array            # [low/high,denominator,numerator] uint8
    fallback_primary: jax.Array    # four move slots, five absent switch slots
    secondary: jax.Array           # move versus switch
    role: jax.Array
    slots: jax.Array
    category_count: jax.Array
    vocabulary_binding: jax.Array
    binding: jax.Array


@dataclass(frozen=True)
class ActionCatalog:
    names: tuple[str,...]
    ids: Mapping[str,int]
    dex: ShowdownDex
    device: ActionTables


def compile_actions(dex,vocabulary,*,hp_limit=DEFAULT_HP_LIMIT):
    if type(hp_limit) is not int or not 1<=hp_limit<=2048:
        raise ValueError('action HP table limit must be an integer in1..2048')
    # Own a frozen species snapshot as well as the compiled move tables, so a
    # request cannot mix a second dex's types/stats with this catalog's moves.
    species={key:replace(info,types=tuple(info.types),base_stats=MappingProxyType(dict(info.base_stats)))
             for key,info in dex.species.items()}
    for info in species.values():
        if any(not np.isfinite(float(x)) or abs(float(x))>np.finfo(np.float32).max for x in info.base_stats.values()):
            raise ValueError('action species stats must fit finite float32')
    dex=ShowdownDex(MappingProxyType(dict(dex.moves)),MappingProxyType(species),MappingProxyType({}))
    names=('',)+tuple(sorted({normalize_id(n) for n in dex.moves}- {''}))
    nums=np.zeros((len(names),2,5),np.float32);cats=np.zeros((len(names),2,4),np.int32)
    kind=np.zeros(len(names),np.int32)
    for index,name in enumerate(names[1:],1):
        info=dex.move_info(name)
        if info is None:continue
        kind[index]=LOW_HP if info.id in ('reversal','flail') else HIGH_HP if info.id in ('eruption','waterspout') else 0
        for ghost in (0,1):
            effect,chance,cost=resolve_move_effect(info,('Ghost',) if ghost else ())
            values=(resolve_move_base_power(info,None),max(-1.,min(1.,float(info.priority)/5)),
                    float(info.accuracy)/100 if info.accuracy else 1.,min(1.,float(chance)/100),max(0.,min(1.,float(cost))))
            if any(not np.isfinite(x) or abs(x)>np.finfo(np.float32).max for x in values):
                raise ValueError('static action mechanics must fit finite float32')
            nums[index,ghost]=values
            move_type='???' if info.id=='struggle' else info.type
            labels=(f'type:{move_type}',f'move_category:{info.gen3_category}',f'move_priority:{info.priority}',
                    'move_effect:'+effect if effect else '')
            cats[index,ghost]=[_static_category(vocabulary,x) if x else 0 for x in labels]
    hp=np.zeros((2,hp_limit+1,hp_limit+1),np.uint8)
    numerator=np.arange(hp_limit+1,dtype=np.float64)
    for denominator in range(1,hp_limit+1):
        fraction=np.clip(numerator/denominator,0.,1.)
        scaled=(48*fraction).astype(np.int32)
        hp[0,denominator]=np.select([scaled<=1,scaled<=4,scaled<=9,scaled<=16,scaled<=32],
            [200,150,100,80,40],default=20).astype(np.uint8)
        hp[1,denominator]=np.maximum(1,(150*fraction).astype(np.int32)).astype(np.uint8)
    fallback=[_static_category(vocabulary,f'move:slot:{i+1}') for i in range(4)]+[
        _static_category(vocabulary,f'species:slot:{i+1}') for i in range(5)]
    secondary=[_static_category(vocabulary,x) for x in ('action:move','action:switch')]
    role=_static_category(vocabulary,'action')
    slots=[_static_category(vocabulary,f'move_slot:{i+1}') for i in range(4)]+[
        _static_category(vocabulary,f'switch_slot:{i+1}') for i in range(5)]
    vb=private_team.vocabulary_binding(vocabulary)
    payload=dict(schema=SCHEMA,names=names,hp_limit=hp_limit,
        species={k:dict(id=v.id,name=v.name,types=v.types,base_stats=dict(v.base_stats)) for k,v in species.items()},
        numeric=nums.tolist(),categories=cats.tolist(),hp_kind=kind.tolist(),
        hp_sha256=hashlib.sha256(hp.tobytes()).hexdigest(),fallback=fallback,secondary=secondary,
        role=role,slots=slots,vocabulary=np.asarray(vb).tolist())
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    binding=jnp.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32))
    return ActionCatalog(names,MappingProxyType({n:i for i,n in enumerate(names) if i}),dex,
        ActionTables(jnp.asarray(nums),jnp.asarray(cats),jnp.asarray(kind),jnp.asarray(hp),
            jnp.asarray(fallback,jnp.int32),jnp.asarray(secondary,jnp.int32),jnp.asarray(role,jnp.int32),
            jnp.asarray(slots,jnp.int32),jnp.asarray(vocabulary.size,jnp.int32),vb,binding))


class ActionFacts(NamedTuple):
    request: private_requests.ActionRequest
    team: private_team.TeamFacts
    catalog_binding: jax.Array
    move_primary: jax.Array         # four checkpoint labels, never mechanics IDs
    mechanics: jax.Array            # four separately bound dex lookup IDs
    actor_ghost: jax.Array
    hp_known: jax.Array
    hp_exact: jax.Array
    hp_numerator: jax.Array
    hp_denominator: jax.Array


def parse_action_facts(request,*,owner_side,vocabulary,catalog):
    dex=catalog.dex
    if not np.array_equal(np.asarray(catalog.device.vocabulary_binding),np.asarray(private_team.vocabulary_binding(vocabulary))):
        raise ValueError('private action vocabulary differs from frozen catalog')
    packet=private_requests.parse_action_request(request,owner_side=owner_side)
    team=private_team.parse_team_facts(request,owner_side=owner_side,vocabulary=vocabulary,dex=dex)
    own=reference._self_team_from_request(request,f'p{owner_side+1}')
    actor=next((p for p in own if p.active),None)
    info=dex.species_info(actor.species) if actor is not None else None
    ghost=info is not None and any(str(t).lower()=='ghost' for t in info.types)
    hp_known=actor is not None and reference._condition_features(actor.condition).hp_fraction is not None
    hp_exact=False;numerator=0;denominator=1
    if hp_known:
        head=str(actor.condition or '').split()[0]
        if '/' in head:
            a,b=(float(x) for x in head.split('/',1))
        else:a,b=0.,1.
        # The bounded integer lookup is only needed for variable-power moves.
        # Other actions retain the source's general float32 HP projection.
        hp_exact=(np.isfinite(a) and np.isfinite(b) and a.is_integer() and b.is_integer()
                  and b>0 and b<catalog.device.hp_power.shape[1])
        if hp_exact:numerator=int(max(0,min(a,b)));denominator=int(b)
    raw=reference._active_request(request)
    rows=raw.get('moves') if isinstance(raw,Mapping) else None
    rows=rows if isinstance(rows,list) else []
    primary=np.asarray(catalog.device.fallback_primary[:4]).copy()
    mechanics=np.zeros(4,np.int32)
    for index,row in enumerate(rows):
        if not isinstance(row,Mapping):continue
        name=reference._request_move_name(row)
        primary[index]=vocabulary.encode('move:'+name)
        lookup=reference._self_move_mechanics_id(row,name,actor.moves if actor is not None else ())
        key=normalize_id(lookup)
        found=catalog.ids.get(key,0)
        if found==0 and dex.move_info(lookup) is not None:
            raise ValueError('known move lies outside frozen action catalog')
        mechanics[index]=found
    return ActionFacts(packet,team,catalog.device.binding,jnp.asarray(primary),jnp.asarray(mechanics),
        jnp.asarray(ghost),jnp.asarray(hp_known),jnp.asarray(hp_exact),jnp.asarray(numerator,jnp.int32),
        jnp.asarray(denominator,jnp.int32))


class ActionFeatures(NamedTuple):
    numeric: jax.Array
    categorical: jax.Array
    legal: jax.Array
    admitted: jax.Array


def project_actions(facts,tables,*,observer_side):
    """Construct the complete action section from one coherent owning packet."""
    shape=facts.request.owner_side.shape
    def require(array,suffix,dtype):
        if getattr(array,'shape',None)!=shape+suffix or getattr(array,'dtype',None)!=jnp.dtype(dtype):
            raise ValueError('private action fact shape or dtype mismatch')
    for name in ('owner_side','request_id','kind','team_size'):
        require(getattr(facts.request,name),(),jnp.int32)
    require(facts.team.owner_side,(),jnp.int32)
    for name in ('present','wait','force_switch','trapped'):
        require(getattr(facts.request,name),(),jnp.bool_)
    for name in ('team_active','team_fainted_prefix'):
        require(getattr(facts.request,name),(6,),jnp.bool_)
    for name in ('move_present','move_disabled','pp_known'):
        require(getattr(facts.request,name),(4,),jnp.bool_)
    for name in ('pp','max_pp'):
        require(getattr(facts.request,name),(4,),jnp.float32)
    for name in ('present','active','fainted','ability_known','item_known'):
        require(getattr(facts.team,name),(6,),jnp.bool_)
    for name in ('level','gender'):
        require(getattr(facts.team,name),(6,),jnp.int32)
    for name in ('hp_numerator','hp_denominator'):
        require(getattr(facts.team,name),(6,),jnp.float32)
    for name in ('base_stats','actual_stats'):
        require(getattr(facts.team,name),(6,6),jnp.float32)
    require(facts.team.categories,(6,7),jnp.int32)
    for x in (facts.actor_ghost,facts.hp_known,facts.hp_exact):require(x,(),jnp.bool_)
    for x in (facts.hp_numerator,facts.hp_denominator):require(x,(),jnp.int32)
    for x in (facts.move_primary,facts.mechanics):require(x,(4,),jnp.int32)
    require(facts.catalog_binding,(8,),jnp.uint32)
    require(facts.team.vocabulary_binding,(8,),jnp.uint32)
    observer=_int32(observer_side)
    if observer.shape not in ((),shape):raise ValueError('observer requires a scalar or matching batch shape')
    ok=(observer>=0)&(observer<=1)&(observer==facts.request.owner_side)&(observer==facts.team.owner_side)
    ok &= jnp.all(facts.catalog_binding==tables.binding,axis=-1)&jnp.all(facts.team.vocabulary_binding==tables.vocabulary_binding,axis=-1)
    ids=facts.mechanics
    ok &= jnp.all((ids>=0)&(ids<tables.numeric.shape[0]),axis=-1)
    ok &= jnp.all((facts.move_primary>=0)&(facts.move_primary<tables.category_count),axis=-1)
    ok &= jnp.all((facts.team.categories>=0)&(facts.team.categories<tables.category_count),axis=(-1,-2))
    ok &= (facts.request.kind>=0)&(facts.request.kind<=private_requests.RequestKind.UNKNOWN)
    ok &= (facts.request.team_size>=0)&(facts.request.team_size<=6)
    ok &= jnp.all(facts.team.present==(jnp.arange(6)<facts.request.team_size[...,None]),axis=-1)
    ok &= jnp.all(facts.request.team_active==facts.team.active,axis=-1)
    for array in (facts.team.hp_numerator,facts.team.hp_denominator,facts.team.base_stats,facts.team.actual_stats,
                  facts.request.pp,facts.request.max_pp):
        ok &= jnp.all(jnp.isfinite(array),axis=tuple(range(len(shape),array.ndim)))
    active=jnp.argmax(facts.team.active,axis=-1)
    actor_num=jnp.take_along_axis(facts.team.hp_numerator,active[...,None],axis=-1)[...,0]
    actor_den=jnp.take_along_axis(facts.team.hp_denominator,active[...,None],axis=-1)[...,0]
    ok &= ~facts.hp_known|jnp.any(facts.team.active,axis=-1)
    ok &= ~facts.hp_exact|((facts.hp_denominator==actor_den)&(facts.hp_numerator==jnp.clip(actor_num,0,actor_den)))
    safe=jnp.clip(ids,0,tables.numeric.shape[0]-1)
    variable=tables.hp_kind[safe]>0
    need_exact=jnp.any(variable&facts.request.move_present,axis=-1)&facts.hp_known
    hp_bound=(facts.hp_denominator>0)&(facts.hp_denominator<tables.hp_power.shape[1])&(
        facts.hp_numerator>=0)&(facts.hp_numerator<=facts.hp_denominator)
    ok &= ~need_exact|(facts.hp_exact&hp_bound)
    actions=private_requests.action_inputs(facts.request,observer_side=observer)
    team=private_team.project_team(facts.team,observer_side=observer,expected_vocabulary_binding=tables.vocabulary_binding)
    ok &= team.admitted
    targets=actions.switch_targets;valid_target=targets>=0;gather=jnp.clip(targets,0,5)
    switch_numeric=jnp.take_along_axis(team.numeric,gather[...,None],axis=-2)
    switch_cats=jnp.take_along_axis(facts.team.categories,gather[...,None],axis=-2)
    numeric=jnp.zeros(shape+(9,132),jnp.float32)
    switch_values=switch_numeric[...,jnp.asarray(SWITCH_COLUMNS)]
    numeric=numeric.at[...,4:,jnp.asarray(SWITCH_COLUMNS)].set(jnp.where(valid_target[...,None],switch_values,0))
    values=tables.numeric[safe,facts.actor_ghost[...,None].astype(jnp.int32)]
    kind=jnp.clip(tables.hp_kind[safe]-1,0,1)
    denominator=jnp.clip(facts.hp_denominator,1,tables.hp_power.shape[1]-1)
    numerator=jnp.clip(facts.hp_numerator,0,tables.hp_power.shape[2]-1)
    dynamic=tables.hp_power[kind,denominator[...,None],numerator[...,None]].astype(jnp.float32)
    power=jnp.where(variable&facts.hp_known[...,None],dynamic,values[...,0])
    values=values.at[...,0].set(jnp.minimum(1,power/200.))
    numeric=numeric.at[...,:4,jnp.asarray(MOVE_COLUMNS)].set(values)
    idx=lambda name:reference.v4_numeric_index(getattr(reference,'NUMERIC_'+name))
    numeric=numeric.at[...,:4,idx('MOVE_PP_FRACTION')].set(actions.move_pp_fraction)
    numeric=numeric.at[...,:4,idx('PRESENT')].set(facts.request.move_present)
    numeric=numeric.at[...,:4,idx('ACTIVE')].set(facts.request.move_present&~facts.request.move_disabled)
    numeric=numeric.at[...,idx('LEGAL')].set(actions.legal)
    categorical=jnp.zeros(shape+(9,41),jnp.int32)
    primary=jnp.concatenate((facts.move_primary,jnp.where(valid_target,switch_cats[...,0],tables.fallback_primary[4:])),axis=-1)
    categorical=categorical.at[...,reference.CATEGORY_PRIMARY].set(primary)
    categorical=categorical.at[...,reference.CATEGORY_SECONDARY].set(jnp.concatenate((jnp.broadcast_to(tables.secondary[0],shape+(4,)),jnp.broadcast_to(tables.secondary[1],shape+(5,))),axis=-1))
    categorical=categorical.at[...,reference.CATEGORY_ROLE].set(tables.role)
    categorical=categorical.at[...,reference.CATEGORY_SLOT].set(tables.slots)
    categories=tables.categories[safe,facts.actor_ghost[...,None].astype(jnp.int32)]
    categorical=categorical.at[...,:4,jnp.asarray(MECHANIC_CATEGORIES)].set(categories)
    categorical=categorical.at[...,4:,reference.CATEGORY_TYPE_1].set(jnp.where(valid_target,switch_cats[...,3],0))
    categorical=categorical.at[...,4:,reference.CATEGORY_TYPE_2].set(jnp.where(valid_target,switch_cats[...,4],0))
    return ActionFeatures(jnp.where(ok[...,None,None],numeric,0),jnp.where(ok[...,None,None],categorical,0),
                          actions.legal&ok[...,None],ok)
