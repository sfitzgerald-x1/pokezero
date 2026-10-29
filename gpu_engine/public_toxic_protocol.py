"""Eight-word raw Toxic controls and canonical shared-event classification.

Counter, known and proof arithmetic belongs exclusively to public_toxic.apply.
The full wire producer must jointly commit these controls with all other raw
consumers. A handled Toxic no-op does not authorize dropping unknown raw kinds.
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from . import public_toxic as toxic

SCHEMA='pokezero.public_toxic_wire.v1'
FLAG_NAMES=('canonical','active_form','tox','residual','baton','retire_all','retire_side')
COMMON_KINDS=(toxic.SWITCH,toxic.DRAG,toxic.REPLACE,toxic.FAINT,toxic.TURN,toxic.UPKEEP,toxic.TEAM)


@dataclass(frozen=True)
class ToxicProjection:
    raw_line: str
    side: int
    payload: tuple[int,...]
    shared_event: bool
    timestamp: bool
    full_line_consumers_complete: bool = False


def project_line(line,*,identity_ids):
    fact=toxic.decode_line(line,identity_ids=identity_ids)
    values={name:np.asarray(x).item() for name,x in zip(fact._fields,fact)}
    flags=sum(int(values[name])<<i for i,name in enumerate(FLAG_NAMES))
    payload=(values['kind'],values['identity'],values['turn'],values['hp'],values['maximum'],flags,0,0)
    shared=(values['kind'] in COMMON_KINDS and values['canonical']) or line=='|'
    return ToxicProjection(line,values['side'],payload,shared,values['kind']==toxic.TIMESTAMP)


def unpack(side,payload):
    """Validate exact device transport before constructing raw semantic facts."""
    if (getattr(payload,'ndim',None)!=2 or payload.shape[1]!=8 or payload.dtype!=jnp.int32
            or getattr(side,'shape',None)!=(payload.shape[0],) or side.dtype!=jnp.int32):
        raise ValueError('Toxic controls require B by eight int32 words and int32 public sides')
    kind,identity,turn,hp,maximum,flags=(payload[:,i] for i in range(6))
    ok=(kind>=toxic.SKIP)&(kind<=toxic.REQUEST)&(side>=-1)&(side<=1)&(identity>=0)&(identity<=6)&(
        turn>=0)&(hp>=-1)&(maximum>=0)&(flags>=0)&(flags<128)&jnp.all(payload[:,6:]==0,axis=-1)
    bits={name:(flags&(1<<i))!=0 for i,name in enumerate(FLAG_NAMES)}
    retire_all=(kind==toxic.TURN)|(kind==toxic.UPKEEP)|(kind==toxic.WIN)|(kind==toxic.TERMINAL)
    retire_side=(kind>=toxic.SWITCH)&(kind<=toxic.FAINT)
    ok &= (kind==toxic.SKIP)|((bits['retire_all']==retire_all)&(bits['retire_side']==retire_side))
    canonical_occupancy=((kind>=toxic.SWITCH)&(kind<=toxic.FAINT))&bits['canonical']
    ok &= ~canonical_occupancy|bits['active_form']
    ok &= ~((kind>=toxic.SWITCH)&(kind<=toxic.REPLACE)&bits['canonical'])|((side>=0)&(identity>0))
    ok &= ~(((kind==toxic.TURN)|(kind==toxic.UPKEEP))&bits['canonical'])|((side==-1)&(identity==0))
    ok &= ~((kind==toxic.TURN)&bits['canonical'])|(turn>=1)
    return toxic.ToxicFact(kind,side,identity,bits['canonical'],bits['active_form'],turn,hp,maximum,
        bits['tox'],bits['residual'],bits['baton'],bits['retire_all'],bits['retire_side']),ok
