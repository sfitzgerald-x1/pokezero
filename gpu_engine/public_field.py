"""Public weather, Spikes and Wish inputs for the champion's retained v4 fields.

This observes protocol events; it does not simulate their battle effects. Sleep
clause and hazard/item credit remain separate writers. Coverage masks are partial.
"""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import NamedTuple, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference
from .private_requests import _KINDS
from .private_team import vocabulary_binding
from .public_mon_state import _int32
from .public_abilities import _bool
from .public_moves import _static_category

SCHEMA = 'pokezero.public_field.v1'
TURN, WEATHER, SPIKES, WISH = range(4)
CORE = ('PRESENT', 'TURN_COUNT', 'SELF_HAZARDS', 'OPP_HAZARDS',
        'WEATHER_TURNS', 'WEATHER_PERMANENT', 'SELF_WISH_PENDING', 'OPP_WISH_PENDING',
        'SELF_WISH_TURNS', 'OPP_WISH_TURNS')
NUMERIC_COLUMNS = tuple(reference.v4_numeric_index(getattr(reference, 'NUMERIC_'+n)) for n in CORE)
CATEGORY_COLUMNS = (reference.CATEGORY_PRIMARY, reference.CATEGORY_SECONDARY, reference.CATEGORY_ROLE)


class FieldTables(NamedTuple):
    weather_category: jax.Array
    request_category: jax.Array
    field_category: jax.Array
    vocabulary_binding: jax.Array
    binding: jax.Array


@dataclass(frozen=True)
class FieldCatalog:
    weather_names: tuple[str, ...]
    weather_ids: Mapping[str, int]
    device: FieldTables


def compile_field(vocabulary, *, weather_names=()):
    names = ('',) + tuple(sorted(({'raindance', 'sunnyday', 'sandstorm', 'hail'} |
        {reference._normalize_identifier(n) for n in weather_names}) - {'', 'none'}))
    weather = [0] + [_static_category(vocabulary, 'weather:'+n) for n in names[1:]]
    requests = [0]*len(_KINDS)
    for name, index in _KINDS.items():
        requests[index] = _static_category(vocabulary, 'request_kind:'+name)
    field = _static_category(vocabulary, 'field')
    vb = vocabulary_binding(vocabulary)
    payload = dict(schema=SCHEMA, names=names, weather=weather, requests=requests,
                   field=field, vocabulary=np.asarray(vb).tolist())
    digest = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    binding = jnp.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32))
    return FieldCatalog(names, MappingProxyType({n:i for i,n in enumerate(names)}),
        FieldTables(jnp.asarray(weather,jnp.int32),jnp.asarray(requests,jnp.int32),
                    jnp.asarray(field,jnp.int32),vb,binding))


class FieldState(NamedTuple):
    binding: jax.Array
    turn: jax.Array
    weather: jax.Array
    weather_set: jax.Array
    weather_ability: jax.Array
    weather_upkeeps: jax.Array
    spikes: jax.Array
    wish_set_turn: jax.Array


def initialize(batch, tables):
    if type(batch) is not int or batch<=0: raise ValueError('positive field batch required')
    z = jnp.zeros(batch,jnp.int32)
    return FieldState(jnp.broadcast_to(tables.binding,(batch,8)),z,z,jnp.zeros(batch,bool),
        jnp.zeros(batch,bool),z,jnp.zeros((batch,2),jnp.int32),jnp.full((batch,2),-1,jnp.int32))


def _batch(state):
    if state.turn.ndim!=1 or not state.turn.shape[0]:
        raise ValueError('field state requires a nonempty batch vector')
    batch=state.turn.shape[0]
    for name in state._fields:
        array=getattr(state,name)
        shape=(batch,8) if name=='binding' else (batch,2) if name in ('spikes','wish_set_turn') else (batch,)
        dtype=jnp.uint32 if name=='binding' else jnp.bool_ if name in ('weather_set','weather_ability') else jnp.int32
        if array.shape!=shape or array.dtype!=dtype:
            raise ValueError('field state shape or dtype mismatch: '+name)
    return batch


def _row(value,batch,*,boolean=False):
    array=_bool(value) if boolean else _int32(value)
    if array.shape not in ((),(batch,)):
        raise ValueError('field row arguments must be scalar or exact batch vectors')
    return array


def apply(state, tables, *, operation, side=-1, value=0, upkeep=False, from_ability=False,
          producer_binding, valid=True):
    """Consume one resolved field fact per battle, refusing invalid rows atomically."""
    if getattr(producer_binding,'shape',None)!=(8,) or getattr(producer_binding,'dtype',None)!=jnp.dtype(jnp.uint32):
        raise ValueError('field producer requires eight uint32 namespace words')
    batch=_batch(state)
    operation,side,value = (_row(x,batch) for x in (operation,side,value))
    upkeep,from_ability,valid = (_row(x,batch,boolean=True) for x in (upkeep,from_ability,valid))
    ok = jnp.all(state.binding==tables.binding,axis=-1)&jnp.all(producer_binding==tables.binding)
    weather = operation==WEATHER
    turn = operation==TURN
    targeted = (operation==SPIKES)|(operation==WISH)
    ok &= (operation>=TURN)&(operation<=WISH)
    ok &= jnp.where(targeted,(side>=0)&(side<=1),side==-1)
    ok &= jnp.where(turn,(state.turn<2**31-1)&(value==state.turn+1),True)
    ok &= jnp.where(weather,(value>=0)&(value<tables.weather_category.shape[0])&
        (~upkeep|(state.weather_upkeeps<2**31-1)),True)
    ok &= jnp.where(targeted,(value==0)|(value==1),True)
    ok &= weather | ~(upkeep|from_ability)
    use = ok&valid
    row = jnp.arange(state.turn.shape[0]); seat = jnp.clip(side,0,1)
    new_turn = jnp.where(use&turn,value,state.turn)
    clear = use&weather&(value==0)
    fresh = use&weather&~upkeep&(value!=0)
    ticks = jnp.where(clear|fresh,0,state.weather_upkeeps+(use&weather&upkeep&(value!=0)).astype(jnp.int32))
    spike = jnp.where(value==0,0,jnp.minimum(3,state.spikes[row,seat]+1))
    wish_old = state.wish_set_turn[row,seat]
    arm = (wish_old<0)|((state.turn-wish_old)>1)
    wish = jnp.where(value==0,-1,jnp.where(arm,state.turn,wish_old))
    return state._replace(turn=new_turn,weather=jnp.where(use&weather,value,state.weather),
        weather_set=jnp.where(clear,False,jnp.where(fresh,True,state.weather_set)),
        weather_ability=jnp.where(clear,False,jnp.where(fresh,from_ability,state.weather_ability)),
        weather_upkeeps=ticks,
        spikes=state.spikes.at[row,seat].set(jnp.where(use&(operation==SPIKES),spike,state.spikes[row,seat])),
        wish_set_turn=state.wish_set_turn.at[row,seat].set(jnp.where(use&(operation==WISH),wish,wish_old))),ok|~valid


@dataclass(frozen=True)
class FieldProjection:
    raw_line: str
    facts: tuple[tuple[int,int,int,bool,bool], ...]
    field_binding: tuple[int, ...]
    full_line_consumers_complete: bool = False


def project_line(line, *, catalog):
    """Decode named field consumers; unhandled lines require another consumer."""
    if not isinstance(line,str) or not line.startswith('|'): raise ValueError('public protocol line required')
    parts=line.split('|');kind=parts[1] if len(parts)>1 else '';facts=[]
    if kind=='request': raise ValueError('private requests cannot enter public field events')
    if kind=='turn':
        number=reference._canonical_turn_number(line)
        if number is None or number>=2**31: raise ValueError('canonical int32 turn required')
        facts.append((TURN,-1,number,False,False))
    elif kind=='-weather':
        name=reference._normalize_identifier(parts[2].strip() if len(parts)>2 else '')
        if name=='none': name=''
        if name not in catalog.weather_ids: raise ValueError('weather outside frozen public namespace')
        facts.append((WEATHER,-1,catalog.weather_ids[name],'[upkeep]' in line,'[from] ability:' in line))
    elif kind in ('-sidestart','-sideend') and len(parts)>=4:
        if reference._side_condition_identifier(parts[3])=='spikes':
            seat=reference._slot_from_ident(parts[2])
            if seat not in ('p1','p2'): raise ValueError('Spikes requires a public side')
            facts.append((SPIKES,int(seat[1])-1,int(kind=='-sidestart'),False,False))
    elif kind=='move' and len(parts)>=4:
        if reference._normalize_identifier(parts[3])=='wish':
            seat=reference._slot_from_ident(parts[2])
            if seat not in ('p1','p2'): raise ValueError('Wish requires a public side')
            facts.append((WISH,int(seat[1])-1,1,False,False))
    elif kind in ('-heal','-sethp') and len(parts)>2 and '[from] move: Wish' in line:
        seat=reference._slot_from_ident(parts[2])
        if seat not in ('p1','p2'): raise ValueError('Wish resolution requires a public side')
        facts.append((WISH,int(seat[1])-1,0,False,False))
    return FieldProjection(line,tuple(facts),tuple(int(x) for x in np.asarray(catalog.device.binding)))


class FieldFeaturePatch(NamedTuple):
    numeric: jax.Array
    categorical: jax.Array
    numeric_columns: jax.Array
    categorical_columns: jax.Array
    admitted: jax.Array


def project(state, tables, request, *, observer_side, expected_vocabulary, exact_state=True):
    """Write retained v4 field columns for a correctly owned private request."""
    batch=_batch(state)
    observer = _row(observer_side,batch)
    exact = _row(exact_state,batch,boolean=True)
    owner=_row(request.owner_side,batch)
    request_kind=_row(request.kind,batch)
    if getattr(expected_vocabulary,'shape',None)!=(8,) or getattr(expected_vocabulary,'dtype',None)!=jnp.dtype(jnp.uint32):
        raise ValueError('field writer requires eight uint32 vocabulary words')
    ok = (observer>=0)&(observer<=1)&(owner==observer)
    ok &= (request_kind>=0)&(request_kind<tables.request_category.shape[0])
    ok &= jnp.all(state.binding==tables.binding,axis=-1)&jnp.all(expected_vocabulary==tables.vocabulary_binding)
    ok &= (state.weather>=0)&(state.weather<tables.weather_category.shape[0])
    row=jnp.arange(state.turn.shape[0]);own=jnp.clip(observer,0,1);opp=1-own
    weather=state.weather>0
    duration=jnp.where(state.weather_ability,5,jnp.where(state.weather_set,jnp.maximum(0,5-state.weather_upkeeps),0))
    def wish(seat):
        start=state.wish_set_turn[row,seat];elapsed=state.turn-start
        pending=(start>=0)&(elapsed<=1)
        remaining=2-elapsed
        turns=jnp.where((start>=0)&(remaining>=1)&(remaining<=2),remaining,0)
        return pending,turns
    own_pending,own_wish=wish(own);opp_pending,opp_wish=wish(opp)
    values=jnp.stack((jnp.ones_like(state.turn),jnp.minimum(1.,state.turn/1000.),
        state.spikes[row,own]/3.,state.spikes[row,opp]/3.,
        jnp.where(exact&weather,duration/5.,0),exact&weather&state.weather_ability,
        exact&own_pending,exact&opp_pending,own_wish/2.,opp_wish/2.),axis=-1).astype(jnp.float32)
    cats=jnp.stack((jnp.broadcast_to(tables.request_category[jnp.clip(request_kind,0,tables.request_category.shape[0]-1)],state.turn.shape),
        tables.weather_category[jnp.clip(state.weather,0,tables.weather_category.shape[0]-1)],
        jnp.broadcast_to(tables.field_category,state.turn.shape)),axis=-1)
    numeric=jnp.zeros((state.turn.shape[0],132),jnp.float32).at[:,jnp.asarray(NUMERIC_COLUMNS)].set(jnp.where(ok[:,None],values,0))
    categorical=jnp.zeros((state.turn.shape[0],41),jnp.int32).at[:,jnp.asarray(CATEGORY_COLUMNS)].set(jnp.where(ok[:,None],cats,0))
    return FieldFeaturePatch(numeric,categorical,jnp.zeros(132,bool).at[jnp.asarray(NUMERIC_COLUMNS)].set(True),
        jnp.zeros(41,bool).at[jnp.asarray(CATEGORY_COLUMNS)].set(True),ok)
