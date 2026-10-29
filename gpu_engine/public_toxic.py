"""Public Toxic counter and bounded reset provenance on device.

The displayed counter is separate from known-zero authorization. Residual
recovery reads the pre-update public HP. Identity words name public encounters,
never private party order. This component does not change physical mechanics.
"""
from typing import NamedTuple
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference
from .private_team import vocabulary_binding
from .public_mon_state import _int32
from .public_abilities import _bool

SCHEMA='pokezero.public_toxic.v1'
SKIP,TIMESTAMP,OTHER,SWITCH,DRAG,REPLACE,FAINT,TURN,UPKEEP,STATUS,CURE,TEAM,CONDITION,MOVE,WIN,TERMINAL,REQUEST=range(17)
UNKNOWN,EXACT,PERCENTAGE=range(3)
COLUMN=reference.v4_numeric_index(reference.NUMERIC_TOXIC_STAGE)


class ToxicTables(NamedTuple):
    species_binding: jax.Array
    vocabulary_binding: jax.Array
    binding: jax.Array


def compile_toxic(species_catalog,vocabulary):
    vb=vocabulary_binding(vocabulary)
    if not np.array_equal(np.asarray(vb),np.asarray(species_catalog.device.vocabulary_binding)):
        raise ValueError('Toxic and public species require the same frozen vocabulary')
    payload=dict(schema=SCHEMA,species=np.asarray(species_catalog.device.binding).tolist(),
        vocabulary=np.asarray(vb).tolist(),column=COLUMN,divisor=15,sentinel=16,identity_capacity=6)
    h=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    return ToxicTables(species_catalog.device.binding,vb,jnp.asarray(np.frombuffer(h,dtype='>u4').astype(np.uint32)))


class ToxicState(NamedTuple):
    binding: jax.Array
    species_binding: jax.Array
    turn: jax.Array
    post_upkeep: jax.Array
    active: jax.Array
    stage: jax.Array
    known: jax.Array
    hp: jax.Array
    maximum: jax.Array
    condition_tox: jax.Array
    visibility: jax.Array
    request_seen: jax.Array
    action_ident: jax.Array
    zero: jax.Array
    zero_ident: jax.Array
    deadline: jax.Array
    pending: jax.Array
    expected: jax.Array
    invalid: jax.Array
    baton: jax.Array


class ToxicFact(NamedTuple):
    kind: jax.Array
    side: jax.Array
    identity: jax.Array
    canonical: jax.Array
    active_form: jax.Array
    turn: jax.Array
    hp: jax.Array
    maximum: jax.Array
    tox: jax.Array
    residual: jax.Array
    baton: jax.Array
    retire_all: jax.Array
    retire_side: jax.Array


def _binding(x):
    if getattr(x,'shape',None)!=(8,) or getattr(x,'dtype',None)!=jnp.dtype(jnp.uint32):
        raise ValueError('Toxic namespace requires eight uint32 words')


def initialize(batch,tables,*,species_binding,complete_prefix=False,hp_visibility=(UNKNOWN,UNKNOWN)):
    if type(batch) is not int or batch<=0 or type(complete_prefix) is not bool:
        raise ValueError('positive batch and explicit boolean complete-prefix provenance required')
    _binding(species_binding)
    if not np.array_equal(np.asarray(species_binding),np.asarray(tables.species_binding)):
        raise ValueError('Toxic state requires its exact public namespace')
    visibility=np.asarray(hp_visibility)
    if visibility.shape not in ((2,),(batch,2)) or visibility.dtype.kind not in 'iu' or np.any((visibility<0)|(visibility>2)):
        raise ValueError('HP visibility requires exact integer unknown/exact/percentage facts')
    z=jnp.zeros((batch,2),jnp.int32);f=jnp.zeros((batch,2),bool)
    return ToxicState(jnp.broadcast_to(tables.binding,(batch,8)),jnp.broadcast_to(species_binding,(batch,8)),
        jnp.zeros(batch,jnp.int32),jnp.zeros(batch,bool),z,z,jnp.full((batch,2),complete_prefix,bool),
        jnp.full((batch,2),-1,jnp.int32),z,f,jnp.broadcast_to(jnp.asarray(visibility,jnp.int32),(batch,2)),
        f,z,f,z,z,f,z,f,f)


def _shape(state,tables):
    if state.turn.ndim!=1 or not state.turn.shape[0]:raise ValueError('nonempty Toxic batch required')
    b=state.turn.shape[0]
    booleans=('post_upkeep','known','condition_tox','request_seen','zero','pending','invalid','baton')
    for name in state._fields:
        shape=(b,8) if name in ('binding','species_binding') else (b,) if name in ('turn','post_upkeep') else (b,2)
        dtype=jnp.uint32 if name in ('binding','species_binding') else jnp.bool_ if name in booleans else jnp.int32
        x=getattr(state,name)
        if x.shape!=shape or x.dtype!=dtype:raise ValueError('Toxic state shape/dtype: '+name)
    for x in tables:_binding(x)
    return b


def _coherent(state,tables):
    ok=jnp.all(state.binding==tables.binding,axis=-1)&jnp.all(state.species_binding==tables.species_binding,axis=-1)&(state.turn>=0)
    for x in (state.active,state.action_ident,state.zero_ident,state.expected):ok &= jnp.all((x>=0)&(x<=6),axis=-1)
    ok &= jnp.all((state.stage>=0)&(state.stage<=16)&(state.hp>=-1)&(state.maximum>=0)&(state.deadline>=0)&
        (state.visibility>=UNKNOWN)&(state.visibility<=PERCENTAGE),axis=-1)
    return ok


def _sanitize(s):
    active=s.active>0
    action=(s.action_ident==s.active)&active&s.condition_tox&s.known&(s.stage==0)
    proof=s.zero&(s.deadline>=1)&(s.zero_ident==s.active)&active&s.condition_tox&s.known&(s.stage==0)&(
        s.deadline==s.turn[:,None]+s.post_upkeep[:,None].astype(jnp.int32))
    pending=s.pending&(s.expected==s.active)&active
    empty_proof=~s.zero&(s.deadline==0)&(s.zero_ident==0)
    empty_pending=~s.pending&(s.expected==0)
    good=(proof|empty_proof)&(pending|empty_pending)&~(s.zero&s.pending)&~(s.invalid&(s.zero|s.pending))
    return s._replace(action_ident=jnp.where(action,s.action_ident,0),zero=s.zero&good,
        zero_ident=jnp.where(good,s.zero_ident,0),deadline=jnp.where(good,s.deadline,0),
        pending=s.pending&good,expected=jnp.where(good,s.expected,0),invalid=s.invalid|~good)


def apply(state,tables,fact,*,producer_binding,valid=True):
    """Consume one complete raw-line fact; malformed boundaries retire proofs.

    canonical/active_form are literal syntax facts. Unknown identity0 is never
    a matching occupant; a canonical arrival requires a registered identity.
    Returning False rejects the whole batch row without partial mutation.
    """
    b=_shape(state,tables);_binding(producer_binding)
    if not isinstance(fact,ToxicFact):raise ValueError('typed raw Toxic fact required')
    flags=('canonical','active_form','tox','residual','baton','retire_all','retire_side')
    data={n:jnp.broadcast_to(_bool(x) if n in flags else _int32(x),(b,)) for n,x in zip(fact._fields,fact)}
    k,side,identity=data['kind'],data['side'],data['identity'];row=jnp.arange(b);seat=jnp.clip(side,0,1)
    named=(side>=0)&(side<2);target=jnp.arange(2)[None,:]==seat[:,None]
    ok=_coherent(state,tables)&jnp.all(producer_binding==tables.binding)&(k>=SKIP)&(k<=REQUEST)&(side>=-1)&(side<=1)&(identity>=0)&(identity<=6)
    ok &= (data['hp']>=-1)&(data['maximum']>=0)&(data['turn']>=0)
    arrive=(k>=SWITCH)&(k<=REPLACE);turn=k==TURN;upkeep=k==UPKEEP;faint=k==FAINT
    status=k==STATUS;cure=k==CURE;team=k==TEAM;condition=k==CONDITION;move=k==MOVE
    ok &= ~(arrive&data['canonical'])|(named&(identity>0))
    ok &= ~((turn)&data['canonical'])|(data['turn']>=1)
    s=_sanitize(state)
    action=s.action_ident
    retire=data['retire_all'][:,None]|(data['retire_side'][:,None]&jnp.where(named[:,None],target,True))
    s=s._replace(action_ident=jnp.where(retire,0,action))
    # The source clears pending replacements before testing the turn marker.
    ordered=data['canonical']&(data['turn']>=1)&(((s.turn==0)&~jnp.any(s.zero,axis=-1))|
        ((s.turn<2**31-1)&(data['turn']==s.turn+1)))
    invalidate_all=(turn&~ordered)|(upkeep&(~data['canonical']|s.post_upkeep))|(arrive&~data['canonical'])
    match=named&data['active_form']&(identity>0)&(identity==s.active[row,seat])
    open_faint=faint&data['canonical']&match&~s.post_upkeep&~s.pending[row,seat]&~s.invalid[row,seat]
    invalidate_all |= faint&~open_faint
    # A non-boundary same-side transition spends a pending faint latch. A raw
    # timestamp returns before this step; it still passes the sanitizer above.
    # REQUEST's side names payload provenance, not a raw protocol ident; its
    # JSON field cannot invalidate a same-seat pending replacement latch.
    generic=~(turn|upkeep|faint|arrive|(k==WIN)|(k==TIMESTAMP)|(k==REQUEST))
    invalidate=invalidate_all[:,None]|(generic[:,None]&named[:,None]&target&s.pending)
    zero=s.zero&~invalidate;zi=jnp.where(invalidate,0,s.zero_ident);deadline=jnp.where(invalidate,0,s.deadline)
    pending=s.pending&~invalidate;expected=jnp.where(invalidate,0,s.expected);invalid=s.invalid|invalidate
    clear_zero=(upkeep&data['canonical'])[:,None]|((turn&ordered)[:,None]&zero&(data['turn'][:,None]!=deadline))|(k==WIN)[:,None]
    zero &= ~clear_zero;zi=jnp.where(clear_zero,0,zi);deadline=jnp.where(clear_zero,0,deadline)
    clear_pending=turn|(k==WIN)
    pending &= ~clear_pending[:,None];expected=jnp.where(clear_pending[:,None],0,expected)
    invalid=jnp.where((turn&ordered)[:,None],False,invalid)
    # The parser's later turn handler re-evaluates ordering AFTER latch
    # invalidation; an initial numbered marker can become admissible here.
    handler_ordered=data['canonical']&(data['turn']>=1)&(((s.turn==0)&~jnp.any(zero,axis=-1))|
        ((s.turn<2**31-1)&(data['turn']==s.turn+1)))
    pending=jnp.where(open_faint[:,None]&target,True,pending)
    expected=jnp.where(open_faint[:,None]&target,identity[:,None],expected)
    replacement=named&arrive&data['canonical']&pending[row,seat]&~invalid[row,seat]&(expected[row,seat]==s.active[row,seat])&(s.active[row,seat]>0)
    pending=jnp.where(arrive[:,None]&named[:,None]&target,False,pending)
    expected=jnp.where(arrive[:,None]&named[:,None]&target,0,expected)
    is_baton=(k==SWITCH)&(s.baton[row,seat]|data['baton'])
    new_zero=(k==SWITCH)&s.post_upkeep&replacement&~is_baton&data['tox']
    ok &= ~new_zero|(s.turn<2**31-1)
    use_arrive=arrive&data['canonical']&named
    at=target&use_arrive[:,None]
    zero=jnp.where(at,new_zero[:,None],zero);zi=jnp.where(at,jnp.where(new_zero,identity,0)[:,None],zi)
    deadline=jnp.where(at,jnp.where(new_zero,s.turn+1,0)[:,None],deadline)
    action=jnp.where(at,jnp.where(((k==SWITCH)|(k==DRAG))&~s.post_upkeep&data['tox'],identity,0)[:,None],s.action_ident)
    stage=jnp.where(at,0,s.stage);known=jnp.where(at,True,s.known)
    active=jnp.where(at,identity[:,None],s.active)
    hp=jnp.where(at,data['hp'][:,None],s.hp);maximum=jnp.where(at,data['maximum'][:,None],s.maximum)
    tox=jnp.where(at,data['tox'][:,None],s.condition_tox)
    # Public poison magnitude is read before the condition update. A missing
    # numerator/denominator never licenses a fabricated counter.
    residual=condition&data['residual']&data['tox']&named
    zero=jnp.where(residual[:,None]&target,False,zero)
    action=jnp.where(residual[:,None]&target,0,action)
    prev=jnp.where(match,s.hp[row,seat],-1);prev_max=jnp.where(match,s.maximum[row,seat],0)
    max_hp=jnp.where(prev_max>0,prev_max,data['maximum']);delta=prev-data['hp']
    recover=residual&(prev>=0)&(data['hp']>0)&(max_hp>0)&(delta>0)
    visibility=s.visibility[row,seat]
    request_visibility=jnp.where(s.request_seen[row,seat],EXACT,jnp.where(jnp.any(s.request_seen,axis=-1),PERCENTAGE,UNKNOWN))
    visibility=jnp.where(max_hp!=100,EXACT,jnp.where(visibility!=UNKNOWN,visibility,request_visibility))
    unit=jnp.maximum(1,max_hp//16);value=delta//unit
    exact=recover&(visibility==EXACT)&(delta%unit==0)&(value>=1)&(value<=15)
    percentage=recover&(visibility==PERCENTAGE)&known[row,seat]&(stage[row,seat]==0)
    unknown=recover&~exact&(visibility!=PERCENTAGE)&(stage[row,seat]==0)
    stage=jnp.where(target&exact[:,None],value[:,None],jnp.where(target&percentage[:,None],1,stage))
    known=jnp.where(target&exact[:,None],True,jnp.where(target&unknown[:,None],False,known))
    update_condition=condition&match
    hp=jnp.where(target&update_condition[:,None],data['hp'][:,None],hp)
    maximum=jnp.where(target&update_condition[:,None],data['maximum'][:,None],maximum)
    tox=jnp.where(target&update_condition[:,None],data['tox'][:,None],tox)
    # Public condition updates require an exact active ident; the source
    # counter helper instead keys its status/cure/faint reset on active syntax.
    cond_status=match&data['active_form']&(status|cure)
    tox=jnp.where(target&cond_status[:,None],status[:,None]&data['tox'][:,None],tox)
    tox=jnp.where(target&team[:,None]&named[:,None],False,tox)
    hp=jnp.where(target&faint[:,None]&match[:,None],-1,hp)
    maximum=jnp.where(target&faint[:,None]&match[:,None],0,maximum)
    tox=jnp.where(target&faint[:,None]&match[:,None],False,tox)
    reset=named&(team|((status|cure|faint)&data['active_form']))
    stage=jnp.where(target&reset[:,None],(status&data['tox']).astype(jnp.int32)[:,None],stage)
    known=jnp.where(target&reset[:,None],True,known)
    zero=jnp.where(target&reset[:,None],False,zero);action=jnp.where(target&reset[:,None],0,action)
    stage=jnp.where((turn&handler_ordered)[:,None]&known&(stage>0),jnp.minimum(16,stage+1),stage)
    baton=jnp.where(target&move[:,None]&named[:,None],data['baton'][:,None],s.baton)
    baton=jnp.where(at|((turn&handler_ordered)[:,None]),False,baton)
    requests=s.request_seen|(target&(k==REQUEST)[:,None]&named[:,None])
    out=s._replace(turn=jnp.where(turn&handler_ordered,data['turn'],s.turn),
        post_upkeep=jnp.where(upkeep&data['canonical'],True,jnp.where(turn&handler_ordered,False,s.post_upkeep)),
        active=active,stage=stage,known=known,hp=hp,maximum=maximum,condition_tox=tox,
        action_ident=action,zero=zero,zero_ident=zi,deadline=deadline,pending=pending,expected=expected,invalid=invalid,baton=baton,request_seen=requests)
    # Truly empty/comment input returns before the source's sanitizer.
    out=jax.tree.map(lambda new,old:jnp.where((k==SKIP).reshape((b,)+(1,)*(old.ndim-1)),old,new),out,state)
    use=ok&jnp.broadcast_to(_bool(valid),(b,))
    out=jax.tree.map(lambda new,old:jnp.where(use.reshape((b,)+(1,)*(old.ndim-1)),new,old),out,state)
    return out,ok|~jnp.broadcast_to(_bool(valid),(b,))


class ToxicPatch(NamedTuple):
    numeric: jax.Array             # [B,2,132], absolute public sides, active-only overlay
    numeric_columns: jax.Array
    admitted: jax.Array


def project_active(state,tables,*,expected_vocabulary):
    """Apply this side patch only where the final candidate.active is true.

    Unknown legacy zero encodes0, but known/proof fields remain separate and
    cannot be reconstructed from this numeric observation.
    """
    b=_shape(state,tables);_binding(expected_vocabulary)
    ok=_coherent(state,tables)&jnp.all(expected_vocabulary==tables.vocabulary_binding)
    values=jnp.minimum(1.,state.stage/15.)
    numeric=jnp.zeros((b,2,132),jnp.float32).at[...,COLUMN].set(jnp.where(ok[:,None],values,0.))
    return ToxicPatch(numeric,jnp.zeros(132,bool).at[COLUMN].set(True),ok)


def decode_line(line,*,identity_ids):
    """Decode raw syntax and integer HP; all counter arithmetic stays on device."""
    if not isinstance(line,str):raise ValueError('raw Toxic line must be text')
    active_words=set()
    for raw_ident,word in identity_ids.items():
        if reference._is_active_protocol_ident(raw_ident):
            key=(reference._slot_from_ident(raw_ident),word)
            if key in active_words:raise ValueError('distinct active idents cannot share one public identity word')
            active_words.add(key)
    stripped=line.strip();parts=stripped.split('|');name=parts[1] if len(parts)>1 else ''
    ident=parts[2] if len(parts)>2 else '';seat=reference._slot_from_ident(ident)
    side=int(seat[1])-1 if seat in ('p1','p2') else -1
    identity=identity_ids.get(ident,0)
    if type(identity) is not int or not 0<=identity<=6:raise ValueError('public Toxic identity must be a registered encounter word')
    kinds={'t:':TIMESTAMP,'switch':SWITCH,'drag':DRAG,'replace':REPLACE,'faint':FAINT,'turn':TURN,
        'upkeep':UPKEEP,'-status':STATUS,'-curestatus':CURE,'-cureteam':TEAM,'-damage':CONDITION,
        '-heal':CONDITION,'-sethp':CONDITION,'move':MOVE,'win':WIN,'request':REQUEST}
    kind=kinds.get(name,TERMINAL if name in reference._TOXIC_RESET_PROOF_BOUNDARY_EVENTS else OTHER)
    if not stripped or stripped.startswith('>'):kind=SKIP
    canonical=True;turn=0;condition='';tox=False;baton=False
    if kind==TURN:
        number=reference._canonical_turn_number(line);canonical=number is not None;turn=number or 0
    elif kind==UPKEEP:canonical=reference._canonical_upkeep_marker(line)
    elif kind==FAINT:canonical=reference._canonical_faint_marker(line,parts)
    elif kind in (SWITCH,DRAG,REPLACE):
        canonical=reference._canonical_replacement_marker(line,name,parts)
        condition=parts[4] if len(parts)>4 else '';baton=reference._line_mentions_baton_pass(parts)
    elif kind==CONDITION:
        if len(parts)<4:kind=OTHER
        else:condition=parts[3]
    elif kind==STATUS:
        if len(parts)<4:kind=OTHER
        else:tox=reference._normalize_identifier(parts[3])=='tox'
    elif kind==MOVE:
        if len(parts)<4:kind=OTHER
        else:baton=reference._normalize_identifier(parts[3])=='batonpass'
    elif kind==REQUEST:
        payload=reference._decode_request_payload(stripped)
        seat=payload.get('side',{}).get('id') if isinstance(payload.get('side'),dict) else None
        side=int(seat[1])-1 if seat in ('p1','p2') else -1
    hp,maximum=reference._hp_numerator_denominator(condition)
    if (kind==CONDITION or kind in (SWITCH,DRAG,REPLACE) and canonical) and not (
            hp is not None and maximum is not None or condition.split()==['0','fnt']):
        raise ValueError('Toxic HP facts require an unsigned integer fraction or literal faint condition')
    hp=-1 if hp is None else hp;maximum=maximum or 0
    if condition:tox='tox' in condition.split()
    residual=name=='-damage' and len(parts)>=4 and any(x.strip()=='[from] psn' for x in parts[4:])
    if any(x>=2**31 for x in (turn,hp,maximum)):raise ValueError('public Toxic integers exceed exact int32 transport')
    raw=(kind,side,identity,canonical,reference._is_active_protocol_ident(ident),turn,hp,maximum,tox,residual,baton,
        name in reference._TOXIC_RESET_PROOF_BOUNDARY_EVENTS,name in reference._TOXIC_RESET_PROOF_OCCUPANCY_EVENTS)
    flags={3,4,8,9,10,11,12}
    return ToxicFact(*(jnp.asarray(x,jnp.bool_ if i in flags else jnp.int32) for i,x in enumerate(raw)))
