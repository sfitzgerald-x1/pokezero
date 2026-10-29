"""Eight-word lexical transport for the public tendency device reducer.

This codec deliberately does not authorize any raw main event consumer.
Ordered stream integration must bind its namespace and consume each line once.
"""
from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np

from . import public_tendency as t


@dataclass(frozen=True)
class TendencyProjection:
    raw_line: str
    side: int
    payload: tuple
    full_line_consumers_complete: bool = False


def project_line(line,*,catalog):
    f=t.decode_line(line,catalog=catalog)
    flags=sum(int(np.asarray(getattr(f,name)))<<i for i,name in enumerate(('called','locked','recharge','baton','ability')))
    payload=tuple(int(np.asarray(getattr(f,name))) for name in ('kind','species','action','target','turn'))
    return TendencyProjection(line,int(np.asarray(f.side)),(*payload,flags,int(np.asarray(f.weather)),int(np.asarray(f.outcome))))


def unpack(payload,side):
    """Return facts and per-row validity; a consumer must refuse invalid rows."""
    if getattr(payload,'ndim',None)!=2 or payload.shape[1]!=8 or payload.dtype!=jnp.int32:
        raise ValueError('tendency payload must be B by eight int32 words')
    if getattr(side,'shape',None)!=(payload.shape[0],) or side.dtype!=jnp.int32:
        raise ValueError('tendency side must be B int32')
    k,species,action,target,turn,flags,weather,outcome=(payload[:,i] for i in range(8))
    ok=(k>=t.OTHER)&(k<=t.UPKEEP)&(side>=-1)&(side<=1)&(species>=0)&(action>=0)&(target>=-1)&(target<=1)&(turn>=0)
    ok &= (flags>=0)&(flags<32)&(weather>=0)&(weather<=3)&jnp.any(outcome[:,None]==jnp.array([0,1,2,3,4,5,9]),axis=-1)
    called,locked,recharge,baton,ability=tuple((flags&(1<<i))!=0 for i in range(5))
    # Kind-specific unused words are canonical zero/empty facts, so bit flips
    # cannot silently acquire another raw event's semantics.
    arrive=(k>=t.SWITCH)&(k<=t.REPLACE)
    ok &= jnp.where(arrive,species>0,species==0)
    ok &= jnp.where((k==t.MOVE)|(k==t.PREPARE),action>0,action==0)
    ok &= jnp.where(k==t.TURN,turn>0,turn==0)
    ok &= (~called&~locked)|(k==t.MOVE)
    ok &= ~recharge|(k==t.CANT)
    ok &= ~baton|arrive
    ok &= ~ability|(k==t.WEATHER)
    ok &= (weather==0)|(k==t.WEATHER)
    ok &= (outcome==9)|(k==t.OUTCOME)
    ok &= (target==-1)|(k==t.MOVE)|(k==t.OUTCOME)
    ok &= ~((k==t.MOVE)&(called&locked))
    ok &= (k!=t.MOVE)|((side>=0)&(target>=0))
    ok &= (k!=t.OUTCOME)|((side>=0)&(target==side)&(outcome<9))
    named=arrive|(k==t.MOVE)|(k==t.CANT)|(k==t.PREPARE)|(k==t.FAINT)|(k==t.PURSUIT)
    ok &= ~named|(side>=0)
    return t.TendencyFact(k,side,species,action,target,turn,called,locked,recharge,baton,weather,ability,outcome),ok
