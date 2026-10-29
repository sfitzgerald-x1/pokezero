"""Lexical pending-Mud-Shot facts for the ordered raw producer.

Resolution is scheduled before a move's actor upsert or after upkeep item
exclusions. The queue flag takes effect only after that move's actor upsert.
This packet is bookkeeping and cannot admit an unknown main raw family.
"""
from dataclasses import dataclass
from typing import NamedTuple
import jax.numpy as jnp
from pokezero.dex import normalize_id
from pokezero.showdown import _slot_from_ident
from . import public_mudshot as mudshot
from .public_mon_state import _int32
from .public_nonproc_protocol import _source_target

NOTE=-1


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
    queue: object
    tagged: object


def project_line(line,before):
    from .public_wire import WireState
    if not isinstance(before,WireState):raise ValueError('Mud Shot fact requires registered public before-state')
    parts=line.split('|');kind=parts[1] if len(parts)>1 else ''
    if kind=='request':raise ValueError('private request is not a public Mud Shot fact')
    if kind=='upkeep' and line!='|upkeep':raise ValueError('Mud Shot upkeep requires the canonical raw line')
    operation={'move':mudshot.RESOLVE,'upkeep':mudshot.RESOLVE,'-damage':mudshot.DAMAGE,
        '-unboost':mudshot.UNBOOST,'faint':mudshot.FAINT}.get(kind,NOTE)
    queue=kind=='move' and len(parts)>3 and normalize_id(parts[3])=='mudshot'
    side=slot=-1;encounter=0
    if queue:
        actor=_slot_from_ident(parts[2])
        if actor not in ('p1','p2'):raise ValueError('Mud Shot move requires a public actor')
        owner=1-(int(actor[1])-1)
        if before.live[owner] and before.active[owner]>=0:
            address=before.encounters[owner][before.active[owner]].address
            side,slot,encounter=address.side,address.slot,address.encounter
    elif operation in (mudshot.DAMAGE,mudshot.UNBOOST,mudshot.FAINT):
        if len(parts)<3:raise ValueError('Mud Shot target ident required')
        address=_source_target(before,parts[2]);side,slot,encounter=address.side,address.slot,address.encounter
    tagged=operation==mudshot.DAMAGE and '[from]' in line
    payload=(operation,side,slot,encounter,int(queue),int(tagged),0,0)
    return Projection(line,payload)


def unpack(payload):
    value=_int32(payload)
    if value.ndim!=2 or value.shape[1]!=8 or not value.shape[0]:raise ValueError('Mud Shot payload requires nonempty B by8 int32 words')
    op,side,slot,enc,queue,tagged=(value[:,i] for i in range(6))
    none=(side==-1)&(slot==-1)&(enc==0)
    target=(side>=0)&(side<2)&(slot>=0)&(slot<6)&(enc>0)
    event=(op==mudshot.DAMAGE)|(op==mudshot.UNBOOST)|(op==mudshot.FAINT)
    ok=(op==NOTE)|event|(op==mudshot.RESOLVE)
    ok &= (queue>=0)&(queue<=1)&(tagged>=0)&(tagged<=1)&jnp.all(value[:,6:]==0,axis=-1)
    ok &= (queue==0)|(op==mudshot.RESOLVE)
    ok &= (tagged==0)|(op==mudshot.DAMAGE)
    ok &= jnp.where(event,target,jnp.where(queue==1,target|none,none))
    return Facts(op,side,slot,enc,queue==1,tagged==1),ok


def words_ok(payload):
    if len(payload)!=8:return False
    op,side,slot,enc,queue,tagged,a,b=payload
    event=op in (mudshot.DAMAGE,mudshot.UNBOOST,mudshot.FAINT)
    none=(side,slot,enc)==(-1,-1,0);target=0<=side<2 and 0<=slot<6 and enc>0
    return (op in (NOTE,mudshot.DAMAGE,mudshot.UNBOOST,mudshot.FAINT,mudshot.RESOLVE)
        and queue in (0,1) and tagged in (0,1) and a==b==0
        and (not queue or op==mudshot.RESOLVE) and (not tagged or op==mudshot.DAMAGE)
        and (target if event else target or none if queue else none))
