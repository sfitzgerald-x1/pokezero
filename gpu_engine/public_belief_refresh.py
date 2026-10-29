"""Ordered source-summary callbacks for actual move evidence updates.

This controller retains the source's first selected move and Choice Band
exclusion. It runs each summary callback at its original evidence basis, before
and after the real device move reducer. Arrival, proc, residual and other raw
consumers still need to invoke their own ordered callbacks in the stream fold.
It does not infer callback multiplicity from a final hard-mask difference.
"""
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import public_belief as belief, public_belief_snapshot as snapshot
from . import public_moves as moves, public_mon_state as mon


class RefreshTables(NamedTuple):
    choice_predicate: jax.Array
    sleep_usable: jax.Array
    source_binding: jax.Array
    move_binding: jax.Array
    binding: jax.Array


def compile_refresh(source, move_catalog):
    _namespace(source.device.binding,(8,));_namespace(move_catalog.device.binding,(8,))
    _namespace(move_catalog.device.source_binding,(8,))
    if not np.array_equal(np.asarray(source.device.binding),np.asarray(move_catalog.device.source_binding)):
        raise ValueError('summary callbacks require the exact move/source namespace')
    choice=source.predicate_id('item','choiceband')
    usable=[name in ('sleeptalk','snore') for name in move_catalog.names]
    payload=dict(schema='pokezero.public_belief_refresh.v1',choice=choice,sleep_usable=usable,
        source=np.asarray(source.device.binding).tolist(),moves=np.asarray(move_catalog.device.binding).tolist())
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    return RefreshTables(jnp.asarray(choice,jnp.int32),jnp.asarray(usable,bool),source.device.binding,
        move_catalog.device.binding,jnp.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32)))


class RefreshState(NamedTuple):
    binding: jax.Array
    visible: snapshot.VisibleSummary
    first_selected: jax.Array      # [B,2 absolute sides,6 public encounters]
    choice_excluded: jax.Array     # persistent conclusion, not reset on entry


class MoveRefreshResult(NamedTuple):
    refresh: RefreshState
    move_update: moves.MoveUpdate
    beliefs: belief.BeliefState
    basis: belief.BeliefState
    admitted: jax.Array
    lifecycle: object = None


def _namespace(value,shape):
    if getattr(value,'shape',None)!=shape or value.dtype!=jnp.uint32:
        raise ValueError('summary callback namespace requires exact uint32 shape '+str(shape))


def _tables(tables,source_tables=None,move_tables=None):
    for name in ('source_binding','move_binding','binding'):_namespace(getattr(tables,name),(8,))
    if tables.choice_predicate.shape!=() or tables.choice_predicate.dtype!=jnp.int32:
        raise ValueError('summary callback choice predicate requires scalar int32')
    if tables.sleep_usable.ndim!=1 or tables.sleep_usable.dtype!=jnp.bool_:
        raise ValueError('summary callback sleep-usable table requires boolean move rows')
    if source_tables is not None:_namespace(source_tables.binding,(8,))
    if move_tables is not None:
        _namespace(move_tables.binding,(8,))
        if tables.sleep_usable.shape!=move_tables.predicate.shape:
            raise ValueError('summary callback move table shape mismatch')


def _state_shape(state,batch):
    _namespace(state.binding,(batch,8))
    for name,dtype in (('first_selected',jnp.int32),('choice_excluded',jnp.bool_)):
        value=getattr(state,name)
        if value.shape!=(batch,2,6) or value.dtype!=dtype:
            raise ValueError('summary callback state shape/dtype: '+name)


def initialize(beliefs,source_tables,tables):
    _tables(tables,source_tables)
    if beliefs.species.ndim!=3 or beliefs.species.shape[1:]!=(2,6):
        raise ValueError('ordered callbacks require B by two observers by six encounters')
    if not np.array_equal(np.asarray(source_tables.binding),np.asarray(tables.source_binding)):
        raise ValueError('summary callback source binding mismatch')
    batch=beliefs.species.shape[0]
    return RefreshState(jnp.broadcast_to(tables.binding,(batch,8)),snapshot.initialize(beliefs,source_tables),
        jnp.zeros((batch,2,6),jnp.int32),jnp.zeros((batch,2,6),bool))


def _choose(ok,old,new):
    return jnp.where(ok.reshape(ok.shape+(1,)*(old.ndim-ok.ndim)),new,old)


def _cache_commit(ok,old,new):
    # Cache namespace words are global; all remaining leaves carry a batch.
    names=('species','source_available','hard','has_evidence','generation','summary')
    return old._replace(**{name:jax.tree.map(lambda a,b:_choose(ok,a,b),
        getattr(old,name),getattr(new,name)) for name in names})


def reset_stay(state,mons,tables,*,side,slot,encounter,valid=True):
    """Entry clears the selected-move lock; the certain exclusion persists."""
    _tables(tables)
    _state_shape(state,mons.species.shape[0])
    rows,seat,target,ok=mon._address(mons,side,slot)
    encounter=mon._int32(encounter);valid=moves._bool(valid)
    ok &= (mons.encounter[rows,seat,target]==encounter)&(encounter>0)
    ok &= (mons.active[rows,seat]==target)&~mons.fainted[rows,seat,target]
    ok &= jnp.all(state.binding==tables.binding,axis=-1)
    return state._replace(first_selected=mon._write(state.first_selected,(rows,seat,target),0,ok&valid)),ok|~valid


def record_move(state,move_state,basis,full_beliefs,mons,source_tables,move_tables,tables,*,
        side,slot,encounter,move_id,caller=moves.NONE,has_from=False,transformed=False,
        opposing_pressure=False,wire_target_side=-1,item_mutated,item_revealed,sleep_pending,
        item_belief_narrowing=False,lifecycle=None,lifecycle_tables=None,public_sleep=None,public_abilities=None,valid=True):
    """Source upsert -> sleep skip -> Choice Band -> PP -> move reveal.

    ``basis`` retains nonreplaceable evidence; ``full_beliefs`` also includes
    current original-ability/item matching. Each new exclusion/reveal applies
    to both. The owning raw consumers supply their *pre-move* public item and
    sleep ledgers, never a precomputed callback count or a Python summary.
    The caller commits these results with its other raw consumers atomically.
    """
    batch=mons.species.shape[0]
    if (lifecycle is None)!=(lifecycle_tables is None):
        raise ValueError('move lifecycle ledger and tables must be supplied together')
    if lifecycle is not None and (public_sleep is None or public_abilities is None):
        raise ValueError('move lifecycle stage requires its pre-move public ledgers')
    previous_lifecycle=lifecycle
    _tables(tables,source_tables,move_tables)
    _state_shape(state,batch)
    for name,value in (('item_mutated',item_mutated),('item_revealed',item_revealed),('sleep_pending',sleep_pending)):
        if getattr(value,'shape',None)!=(batch,2,6) or value.dtype!=jnp.bool_:
            raise ValueError('ordered move callbacks require exact boolean public '+name+' facts')
    if basis.species.shape!=(batch,2,6) or full_beliefs.species.shape!=basis.species.shape:
        raise ValueError('ordered move callback belief shape')
    if state.first_selected.shape!=(batch,2,6) or state.first_selected.dtype!=jnp.int32:
        raise ValueError('ordered move callback selected-move shape/dtype')
    if state.choice_excluded.shape!=(batch,2,6) or state.choice_excluded.dtype!=jnp.bool_:
        raise ValueError('ordered move callback exclusion shape/dtype')
    valid=moves._bool(valid);narrow=moves._bool(item_belief_narrowing)
    rows,seat,target,address_ok=mon._address(mons,side,slot)
    mid=jnp.broadcast_to(mon._int32(move_id),(batch,));caller=jnp.broadcast_to(mon._int32(caller),(batch,))
    safe=jnp.clip(mid,0,move_tables.predicate.shape[0]-1)
    transformed=moves._bool(transformed)
    # Validate the actual PP/reveal transaction first, without committing it.
    probe=moves.record_move(move_state,basis,mons,move_tables,source_tables,
        side=side,slot=slot,encounter=encounter,move_id=mid,caller=caller,has_from=has_from,
        transformed=transformed,opposing_pressure=opposing_pressure,wire_target_side=wire_target_side,valid=valid)
    ok=probe.admitted&address_ok
    ok &= jnp.all(state.binding==tables.binding,axis=-1)
    ok &= jnp.all(tables.source_binding==source_tables.binding)&jnp.all(tables.move_binding==move_tables.binding)
    ok &= jnp.all(full_beliefs.species==basis.species,axis=(1,2))
    ok &= jnp.all(full_beliefs.source_available==basis.source_available,axis=(1,2))
    ok &= jnp.all(full_beliefs.pin==basis.pin,axis=(1,2,3))
    ok &= jnp.all(full_beliefs.pin_active==basis.pin_active,axis=(1,2))
    ok &= jnp.all(full_beliefs.conflicts==basis.conflicts,axis=(1,2))
    ok &= jnp.all((full_beliefs.hard&basis.hard)==full_beliefs.hard,axis=(1,2,3))
    ok &= jnp.all(~basis.has_evidence|full_beliefs.has_evidence,axis=(1,2))
    _,cache_ok=snapshot.read(state.visible,full_beliefs,source_tables)
    ok &= jnp.all(cache_ok,axis=(1,2))
    ok &= (tables.choice_predicate>=0)&(tables.choice_predicate<source_tables.predicates.shape[0])
    ok &= jnp.all((state.first_selected>=0)&(state.first_selected<move_tables.predicate.shape[0]),axis=(1,2))
    first=state.first_selected[rows,seat,target]
    ok &= (first>=0)&(first<move_tables.predicate.shape[0])
    selected=probe.revealed&(caller!=moves.SLEEP_TALK)
    exclude=selected&narrow&(first>0)&(first!=mid)
    exclude &= ~item_mutated[rows,seat,target]&~item_revealed[rows,seat,target]&~state.choice_excluded[rows,seat,target]
    skip=(caller==moves.NONE)&tables.sleep_usable[safe]&sleep_pending[rows,seat,target]
    mask=(jnp.arange(2)[None,:,None]==(1-seat)[:,None,None])&(jnp.arange(6)[None,None,:]==target[:,None,None])
    current,base,visible=full_beliefs,basis,state.visible
    def callback(current,base,visible,when,ok):
        visible,current,accepted=snapshot.refresh(visible,current,source_tables,
            source_refresh=mask&(when&ok)[:,None,None])
        ok &= jnp.all(accepted,axis=(1,2))
        return current,base._replace(conflicts=current.conflicts),visible,ok
    current,base,visible,ok=callback(current,base,visible,valid,ok) # _upsert, even called/transformed
    current,base,visible,ok=callback(current,base,visible,skip,ok)
    if lifecycle is not None:
        # The source increments cure epochs here, before called/transformed
        # early returns and before choice/PP/reveal replacements.
        from . import public_belief_lifecycle as life
        effect=life.apply(lifecycle,public_sleep,public_abilities,mons,lifecycle_tables,
            operation=life.MOVE,side=side,slot=slot,encounter=encounter,move_id=mid,
            producer_binding=lifecycle_tables.binding,valid=valid&ok)
        lifecycle=effect.state;ok &= effect.admitted
    use=mask&(exclude&ok)[:,None,None]
    for which in ('base','current'):
        before=base if which=='base' else current
        after,accepted=belief.add_evidence(before,source_tables,tables.choice_predicate,exclude=True,valid=use)
        ok &= jnp.all(accepted,axis=(1,2))
        if which=='base':base=after
        else:current=after
    current,base,visible,ok=callback(current,base,visible,exclude,ok)
    current,base,visible,ok=callback(current,base,visible,probe.charge>0,ok)
    update=moves.record_move(move_state,base,mons,move_tables,source_tables,
        side=side,slot=slot,encounter=encounter,move_id=mid,caller=caller,has_from=has_from,
        transformed=transformed,opposing_pressure=opposing_pressure,wire_target_side=wire_target_side,valid=valid&ok)
    base=update.beliefs
    current,_=belief.add_evidence(current,source_tables,move_tables.predicate[safe][:,None,None],
        valid=mask&(update.revealed&ok)[:,None,None])
    current,base,visible,ok=callback(current,base,visible,update.revealed,ok)
    apply=valid&ok
    first_selected=mon._write(state.first_selected,(rows,seat,target),mid,apply&selected&(first==0))
    excluded=mon._write(state.choice_excluded,(rows,seat,target),True,apply&exclude)
    out=state._replace(visible=_cache_commit(apply,state.visible,visible),first_selected=first_selected,choice_excluded=excluded)
    choose=lambda a,b:_choose(apply,a,b)
    result_beliefs=jax.tree.map(choose,full_beliefs,current)
    result_base=jax.tree.map(choose,basis,base)
    move_update=update._replace(state=jax.tree.map(choose,move_state,update.state),beliefs=result_base,
        admitted=apply,revealed=apply&update.revealed,charge=jnp.where(apply,update.charge,0),
        doubled_despite_wire=apply&update.doubled_despite_wire)
    if lifecycle is not None:
        lifecycle=jax.tree.map(choose,previous_lifecycle,lifecycle)
    return MoveRefreshResult(out,move_update,result_beliefs,result_base,ok|~valid,lifecycle)
