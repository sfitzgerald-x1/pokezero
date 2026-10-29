"""Distinct belief and public-parser Sleep Clause ledgers for v4 field rows.

Consumes resolved public status/cure/faint facts, with the previous sleep state
at the same source boundary. The belief holder excludes ability-induced sleep;
the parser victim set intentionally does not. No private opponent request or
sampled physical state is an input. Raw-fold integration is a separate step.
"""
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference
from . import public_sleep as sleep, public_mon_state as mon
from .public_abilities import _bool
from .private_team import TeamFeaturePatch

ARRIVE,STATUS,CURE,TEAM,FAINT=range(5)
COLUMNS=tuple(reference.v4_numeric_index(getattr(reference,'NUMERIC_'+name)) for name in
    ('SELF_SLEEP_CLAUSE','OPP_SLEEP_CLAUSE','SLEEP_CLAUSE_BLOCKS_SELF','SLEEP_CLAUSE_BLOCKS_OPP'))


class Tables(NamedTuple):
    binding: jax.Array
    sleep_binding: jax.Array
    species_binding: jax.Array
    vocabulary_binding: jax.Array


class State(NamedTuple):
    binding: jax.Array
    encounter: jax.Array
    holder: jax.Array       # B,2: inducing side -> opponent's encounter slot, -1 absent
    victims: jax.Array      # B,2,6: victim side and resolved public encounter slot
    complete: jax.Array


def compile_clause(sleep_tables):
    for value in (sleep_tables.binding,sleep_tables.species_binding,sleep_tables.vocabulary_binding):sleep._binding(value)
    blob=json.dumps(dict(schema='pokezero.public_sleep_clause.v1',columns=COLUMNS,
        sleep=np.asarray(sleep_tables.binding).tolist(),species=np.asarray(sleep_tables.species_binding).tolist(),
        vocabulary=np.asarray(sleep_tables.vocabulary_binding).tolist()),sort_keys=True,separators=(',',':')).encode()
    binding=jnp.asarray(np.frombuffer(hashlib.sha256(blob).digest(),dtype='>u4').astype(np.uint32))
    return Tables(binding,sleep_tables.binding,sleep_tables.species_binding,sleep_tables.vocabulary_binding)


def initialize(batch,tables,*,complete_prefix):
    if type(batch) is not int or batch<=0 or type(complete_prefix) is not bool:
        raise ValueError('positive batch and explicit complete-prefix fact required')
    return State(jnp.broadcast_to(tables.binding,(batch,8)),jnp.zeros((batch,2,6),jnp.int32),
        jnp.full((batch,2),-1,jnp.int32),jnp.zeros((batch,2,6),bool),jnp.full(batch,complete_prefix,bool))


def _shape(state,tables):
    if state.complete.ndim!=1 or not state.complete.shape[0]:raise ValueError('nonempty clause batch required')
    b=state.complete.shape[0]
    for value,shape,dtype in ((state.binding,(b,8),jnp.uint32),(state.encounter,(b,2,6),jnp.int32),
            (state.holder,(b,2),jnp.int32),(state.victims,(b,2,6),jnp.bool_),(state.complete,(b,),jnp.bool_)):
        if getattr(value,'shape',None)!=shape or getattr(value,'dtype',None)!=jnp.dtype(dtype):
            raise ValueError('clause state shape/dtype mismatch')
    for value in tables:sleep._binding(value)
    return b


def _coherent(state,tables):
    ok=jnp.all(state.binding==tables.binding,axis=-1)&jnp.all(state.encounter>=0,axis=(1,2))
    ok &= jnp.all((state.holder>=-1)&(state.holder<6),axis=1)
    for side in (0,1):
        slot=jnp.clip(state.holder[:,side],0,5)
        present=state.encounter[jnp.arange(state.complete.shape[0]),1-side,slot]>0
        ok &= (state.holder[:,side]<0)|present
    ok &= ~jnp.any(state.victims&(state.encounter==0),axis=(1,2))
    return ok


def validate(state,tables,mons,sleep_state):
    """Check every persistent field and join before any event or empty chunk."""
    b=_shape(state,tables)
    for value,shape,dtype in ((mons.encounter,(b,2,6),jnp.int32),(mons.species_binding,(b,8),jnp.uint32),
            (sleep_state.encounter,(b,2,6),jnp.int32),(sleep_state.binding,(b,8),jnp.uint32),
            (sleep_state.species_binding,(b,8),jnp.uint32)):
        if getattr(value,'shape',None)!=shape or getattr(value,'dtype',None)!=jnp.dtype(dtype):
            raise ValueError('clause entry public-state shape/dtype mismatch')
    ok=_coherent(state,tables)&jnp.all(mons.species_binding==tables.species_binding,axis=-1)
    ok &= jnp.all(sleep_state.binding==tables.sleep_binding,axis=-1)&jnp.all(sleep_state.species_binding==tables.species_binding,axis=-1)
    ok &= jnp.all(state.encounter==mons.encounter,axis=(1,2))&jnp.all(state.encounter==sleep_state.encounter,axis=(1,2))
    return ok


def apply(state,tables,mons,sleep_before,*,operation,side,slot=-1,encounter=0,status=0,
          rest=False,ability_source=False,parser_ident_matches,producer_binding,valid=True):
    """Apply one canonical source fact; the caller commits the whole raw frame."""
    b=_shape(state,tables);rows=jnp.arange(b);sleep._binding(producer_binding)
    def scalar(value,dtype):
        value=mon._int32(value) if dtype==jnp.int32 else _bool(value)
        if value.shape not in ((),(b,)):raise ValueError('clause fact must be scalar or exact batch vector')
        return jnp.broadcast_to(value,(b,))
    op,side,slot,encounter,status=(scalar(v,jnp.int32) for v in (operation,side,slot,encounter,status))
    rest,ability_source,parser_match,enabled=(scalar(v,jnp.bool_) for v in (rest,ability_source,parser_ident_matches,valid))
    for value,shape,dtype in ((mons.encounter,(b,2,6),jnp.int32),(mons.species_binding,(b,8),jnp.uint32),
            (sleep_before.encounter,(b,2,6),jnp.int32),(sleep_before.binding,(b,8),jnp.uint32),
            (sleep_before.species_binding,(b,8),jnp.uint32),(sleep_before.status,(b,2,6),jnp.int32),
            (sleep_before.condition_status,(b,2,6),jnp.int32),(sleep_before.literal_faint,(b,2,6),jnp.bool_)):
        if getattr(value,'shape',None)!=shape or getattr(value,'dtype',None)!=jnp.dtype(dtype):
            raise ValueError('clause prior public-state shape/dtype mismatch')
    team=op==TEAM;arrival=op==ARRIVE
    seat=jnp.clip(side,0,1);target=jnp.clip(slot,0,5)
    ok=_coherent(state,tables)&jnp.all(producer_binding==tables.binding)
    ok &= jnp.all(mons.species_binding==tables.species_binding,axis=-1)
    ok &= jnp.all(sleep_before.binding==tables.sleep_binding,axis=-1)&jnp.all(sleep_before.species_binding==tables.species_binding,axis=-1)
    ok &= jnp.all(state.encounter==sleep_before.encounter,axis=(1,2))
    ok &= jnp.all((sleep_before.status>=mon.NONE)&(sleep_before.status<=mon.TOXIC)&
                  (sleep_before.condition_status>=mon.NONE)&(sleep_before.condition_status<=mon.TOXIC),axis=(1,2))
    ok &= (op>=ARRIVE)&(op<=FAINT)&(side>=0)&(side<2)
    ok &= jnp.where(team,(slot==-1)&(encounter==0),
        (slot>=0)&(slot<6)&(encounter>0)&(mons.encounter[rows,seat,target]==encounter))
    ok &= (status>=mon.NONE)&(status<=mon.TOXIC)
    ok &= jnp.where(op==STATUS,(status>=mon.SLEEP)&(~rest|(status==mon.SLEEP)),
        jnp.where(op==CURE,(status>=mon.SLEEP)&~rest&~ability_source,(status==0)&~rest&~ability_source))
    # Arrival changes only its named encounter; every other row must match the
    # prior ledger. A slot cannot be reused for a different public encounter.
    named=(jnp.arange(2)[None,:,None]==seat[:,None,None])&(jnp.arange(6)[None,None,:]==target[:,None,None])
    ok &= jnp.all(jnp.where(arrival[:,None,None]&named,
        (state.encounter==0)|(state.encounter==mons.encounter),state.encounter==mons.encounter),axis=(1,2))
    encounters=jnp.where(arrival[:,None,None]&named,mons.encounter,state.encounter)
    holders=state.holder;victims=state.victims
    induced=(op==STATUS)&(status==mon.SLEEP)&~rest
    prior=holders[rows,1-seat]
    holders=holders.at[rows,1-seat].set(jnp.where(induced&~ability_source,target,prior))
    victims=victims.at[rows,seat,target].set(victims[rows,seat,target]|induced)
    # Belief cures clear a matching holder even for a non-sleep cure token.
    # Parser victims clear only on slp or faint with the same raw nickname key,
    # even if a benched species/base-forme alias resolved to this encounter.
    # Whole-team parser cures
    # deliberately include fainted victims. Do not merge these two surfaces.
    clear=(op==CURE)|(op==FAINT)
    ok &= clear|~parser_match
    living_status=(~sleep_before.literal_faint[rows,seat])&((sleep_before.status[rows,seat]!=mon.NONE)|
        (sleep_before.condition_status[rows,seat]!=mon.NONE))
    for inducing in (0,1):
        held=jnp.clip(holders[:,inducing],0,5)
        match=(inducing==1-seat)&(holders[:,inducing]==target)
        clear_team=team&(inducing==1-seat)&living_status[rows,held]
        holders=holders.at[:,inducing].set(jnp.where((clear&match)|clear_team,-1,holders[:,inducing]))
    victims=jnp.where((parser_match&((op==FAINT)|((op==CURE)&(status==mon.SLEEP))))[:,None,None]&named,False,victims)
    victims=jnp.where(team[:,None,None]&(jnp.arange(2)[None,:,None]==seat[:,None,None]),False,victims)
    candidate=state._replace(encounter=encounters,holder=holders,victims=victims)
    accepted=enabled&ok
    choose=lambda old,new:jnp.where(accepted.reshape((b,)+(1,)*(old.ndim-1)),new,old)
    return jax.tree.map(choose,state,candidate),~enabled|ok


def project(state,tables,*,observer_side,expected_vocabulary,exact_state=True):
    b=_shape(state,tables);sleep._binding(expected_vocabulary)
    observer=mon._int32(observer_side);exact=_bool(exact_state)
    if observer.shape not in ((),(b,)) or exact.shape not in ((),(b,)):
        raise ValueError('clause observer/mask must be scalar or exact batch vector')
    observer=jnp.broadcast_to(observer,(b,));exact=jnp.broadcast_to(exact,(b,));rows=jnp.arange(b);seat=jnp.clip(observer,0,1)
    ok=_coherent(state,tables)&state.complete&(observer>=0)&(observer<2)&jnp.all(expected_vocabulary==tables.vocabulary_binding)
    blocks=jnp.stack((jnp.any(state.victims[rows,1-seat],axis=-1),jnp.any(state.victims[rows,seat],axis=-1)),axis=-1)
    values=jnp.concatenate((jnp.stack((state.holder[rows,seat]>=0,state.holder[rows,1-seat]>=0),axis=-1)&exact[:,None],blocks),axis=-1)
    numeric=jnp.zeros((b,132),jnp.float32).at[:,jnp.asarray(COLUMNS)].set(jnp.where(ok[:,None],values,False).astype(jnp.float32))
    return TeamFeaturePatch(numeric,jnp.zeros((b,41),jnp.int32),jnp.zeros(132,bool).at[jnp.asarray(COLUMNS)].set(True),jnp.zeros(41,bool),ok)
