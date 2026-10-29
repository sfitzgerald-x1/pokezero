"""Lexical non-proc control facts for subsequent shared-fold registration.

A NOTE is bookkeeping only: this producer cannot admit an unknown main raw
family. Public encounter lookup and integer condition facts are explicit;
no candidate summaries, outcome labels or private requests enter this codec.
"""
from dataclasses import dataclass
from typing import NamedTuple
import jax.numpy as jnp
from . import public_item_nonproc as nonproc
from .public_mon_state import _int32
from pokezero.dex import normalize_id
from pokezero.showdown import _slot_from_ident,_species_from_ident

UPKEEP=3


@dataclass(frozen=True)
class Projection:
    raw_line: str
    payload: tuple[int,...]
    full_line_consumers_complete: bool=False


class Facts(NamedTuple):
    operation: object
    side: object
    slot: object
    encounter: object
    hp: object
    maximum: object
    phase: object
    action_phase_hp: object
    residual_phase_hp: object
    leftovers_heal: object
    berry_eat: object


def _source_target(state,ident):
    side=_slot_from_ident(ident)
    if side not in ('p1','p2'):raise ValueError('non-proc fact requires a public side')
    owner=int(side[1])-1
    if state.live[owner] and state.active[owner]>=0:
        return state.encounters[owner][state.active[owner]].address
    name=normalize_id(_species_from_ident(ident))
    matches=[entry.address for entry in state.encounters[owner] if normalize_id(entry.species)==name]
    if len(matches)!=1:raise ValueError('inactive non-proc lookup requires one literal public species record')
    return matches[0]


def project_line(line,before,after=None):
    from .public_wire import WireState,_arrive
    if not isinstance(before,WireState):raise ValueError('registered public before-state required')
    facts=nonproc.lexical_facts(line);parts=line.split('|')
    operation=UPKEEP if facts.upkeep else nonproc.ARRIVE if facts.arrival else nonproc.HP if facts.hp_line else nonproc.NOTE
    side=slot=-1;encounter=hp=maximum=0
    targeted=facts.arrival or facts.hp_line or facts.leftovers_heal or facts.berry_eat
    if targeted:
        if len(parts)<3:raise ValueError('non-proc target ident required')
        if facts.arrival:
            if not isinstance(after,WireState):raise ValueError('non-proc arrival requires its registered candidate')
            expected,_=_arrive(before,line,parts)
            if (after.encounters!=expected.encounters or after.active!=expected.active or after.live!=expected.live):
                raise ValueError('non-proc arrival candidate differs from this raw public transition')
            for name in before.__dataclass_fields__:
                if name.endswith('_binding') and getattr(before,name) is not None and getattr(after,name)!=getattr(before,name):
                    raise ValueError('non-proc arrival candidate changes a frozen namespace')
            owner=_slot_from_ident(parts[2])
            if owner not in ('p1','p2'):raise ValueError('non-proc arrival requires a public side')
            owner=int(owner[1])-1
            if not after.live[owner] or after.active[owner]<0:raise ValueError('non-proc arrival requires a live public candidate')
            address=after.encounters[owner][after.active[owner]].address
        else:address=_source_target(before,parts[2])
        side,slot,encounter=address.side,address.slot,address.encounter
    if facts.arrival or facts.hp_line:
        from .public_mon_protocol import condition_words
        index=4 if facts.arrival else 3
        if len(parts)<=index:raise ValueError('non-proc public condition required')
        hp,maximum,_,_=condition_words(parts[index])
    action=facts.action_phase_hp if facts.hp_line else nonproc.KEEP
    residual=facts.residual_phase_hp if facts.hp_line else nonproc.KEEP
    word=facts.phase|(action<<2)|(residual<<4)
    flags=int(facts.leftovers_heal)|(int(facts.berry_eat)<<1)
    payload=(operation,side,slot,encounter,hp,maximum,word,flags)
    if any(type(value) is not int or not -(2**31)<=value<2**31 for value in payload):
        raise ValueError('non-proc lexical word exceeds int32')
    return Projection(line,payload)


def unpack(payload):
    value=_int32(payload)
    if value.ndim!=2 or value.shape[1]!=8 or not value.shape[0]:raise ValueError('non-proc payload requires nonempty B by8 int32 words')
    op,side,slot,enc,hp,maximum,word,flags=(value[:,i] for i in range(8))
    phase=word&3;action=(word>>2)&3;residual=(word>>4)&3
    left=(flags&1)!=0;berry=(flags&2)!=0
    targeted=(op==nonproc.ARRIVE)|(op==nonproc.HP)|left|berry
    ok=(op>=nonproc.NOTE)&(op<=UPKEEP)&(word>=0)&(word<64)&(phase<=2)&(action<=2)&(residual<=2)&(flags>=0)&(flags<=3)
    ok &= jnp.where(targeted,(side>=0)&(side<2)&(slot>=0)&(slot<6)&(enc>0),(side==-1)&(slot==-1)&(enc==0))
    ok &= jnp.where((op==nonproc.ARRIVE)|(op==nonproc.HP),(hp>=0)&(maximum>=hp),(hp==0)&(maximum==0))
    ok &= ((op==nonproc.HP)|((action==0)&(residual==0)))&(~left|(op==nonproc.HP))&(~berry|(op==nonproc.NOTE))
    ok &= (op!=nonproc.ARRIVE)|((phase==nonproc.ACTION_PHASE)&(maximum>0))
    ok &= (op!=nonproc.HP)|(phase==nonproc.SAME_PHASE)
    ok &= ~berry|(phase==nonproc.SAME_PHASE)
    ok &= (op!=UPKEEP)|((word==0)&(flags==0))
    return Facts(op,side,slot,enc,hp,maximum,phase,action,residual,left,berry),ok


def words_ok(payload):
    """Host validation without a device dispatch for every recorded event."""
    if len(payload)!=8:return False
    op,side,slot,enc,hp,maximum,word,flags=payload
    phase,action,residual=word&3,(word>>2)&3,(word>>4)&3
    left,berry=bool(flags&1),bool(flags&2)
    targeted=op in (nonproc.ARRIVE,nonproc.HP) or left or berry
    return (nonproc.NOTE<=op<=UPKEEP and 0<=word<64 and phase<=2 and action<=2 and residual<=2 and 0<=flags<=3
        and ((0<=side<2 and 0<=slot<6 and enc>0) if targeted else (side,slot,enc)==(-1,-1,0))
        and ((0<=hp<=maximum) if op in (nonproc.ARRIVE,nonproc.HP) else hp==maximum==0)
        and (op==nonproc.HP or action==residual==0) and (not left or op==nonproc.HP)
        and (not berry or op==nonproc.NOTE)
        and (op!=nonproc.ARRIVE or phase==nonproc.ACTION_PHASE and maximum>0)
        and (op!=nonproc.HP or phase==nonproc.SAME_PHASE) and (not berry or phase==nonproc.SAME_PHASE)
        and (op!=UPKEEP or word==flags==0))
