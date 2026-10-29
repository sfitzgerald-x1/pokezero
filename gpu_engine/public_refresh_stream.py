"""Raw transaction envelope for ordered source-summary consumers.

The first integrated families are arrival, move, turn and public abilities.
Unsupported source callback families refuse at the lexical boundary until their
intermediate evidence producers are installed. This is not replay admission.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp

from pokezero.belief import _trace_copy_source
from pokezero.dex import normalize_id
from . import public_belief_refresh as refresh, public_belief_snapshot as snapshot
from . import public_tendency as tendency

NOOP, ARRIVAL, MOVE, TURN, ABILITY, TRACE, TRANSFORM=range(1,8)
SCHEMA='pokezero.public_refresh_stream.v1'
FAMILIES=('noop','arrival','move','turn','ability','trace','transform')


class StreamState(NamedTuple):
    controller: refresh.RefreshState
    family: jax.Array
    main_seen: jax.Array
    main_proof: jax.Array          # side, tendency kind, action/species, target, flags, turn


def validate_source(source):
    for variants in source.variants:
        abilities={normalize_id(v.ability or '') for v in variants}
        if 'intimidate' in abilities and len(abilities)>1:
            raise ValueError('ordered refresh requires the pending Intimidate source consumer for this source')


def family(line):
    """Lexical source family, never a CPU summary or callback count."""
    parts=line.split('|');kind=parts[1] if len(parts)>1 else ''
    if kind in ('switch','drag','replace'):return ARRIVAL
    if kind=='move':
        if len(parts)<4 or normalize_id(parts[3])=='mudshot':
            raise ValueError('pending MudShot evidence requires its ordered refresh consumer')
        return MOVE
    if kind=='turn':return TURN
    if kind in ('ability','-ability'):return TRACE if _trace_copy_source(line)[0] else ABILITY
    if kind=='-transform':return TRANSFORM
    if line=='|' or kind in ('t:','-weather','-sidestart','-sideend','-boost','-unboost','-setboost',
            '-clearboost','-clearpositiveboost','-clearnegativeboost','-clearallboost','-copyboost',
            '-start','-end','-activate','-immune','-fail','-block','-miss','-crit','-supereffective',
            '-resisted','-anim','-mustrecharge','-singleturn','-singlemove','-prepare'):
        return NOOP
    raise ValueError('raw source summary family still requires its ordered evidence consumer: '+kind)


def initialize(beliefs,source_tables,tables):
    batch=beliefs.species.shape[0]
    return StreamState(refresh.initialize(beliefs,source_tables,tables),jnp.zeros(batch,jnp.int32),
        jnp.zeros(batch,jnp.int32),jnp.zeros((batch,6),jnp.int32))


def validate(state,beliefs,source_tables,tables,move_tables):
    batch=beliefs.species.shape[0]
    refresh._tables(tables,source_tables,move_tables);refresh._state_shape(state.controller,batch)
    for value in (state.family,state.main_seen):
        if value.shape!=(batch,) or value.dtype!=jnp.int32:
            raise ValueError('source refresh frame requires B int32 state words')
    if state.main_proof.shape!=(batch,6) or state.main_proof.dtype!=jnp.int32:
        raise ValueError('source refresh main proof requires B by6 int32 words')
    _,visible_ok=snapshot.read(state.controller.visible,beliefs,source_tables)
    ok=jnp.all(state.controller.binding==tables.binding,axis=-1)
    ok &= jnp.all(tables.source_binding==source_tables.binding)&jnp.all(tables.move_binding==move_tables.binding)
    ok &= (tables.choice_predicate>=0)&(tables.choice_predicate<source_tables.predicates.shape[0])
    selected=state.controller.first_selected
    ok &= jnp.all((selected>=0)&(selected<tables.sleep_usable.shape[0]),axis=(1,2))
    ok &= (state.family>=0)&(state.family<=TRANSFORM)&(state.main_seen>=0)&(state.main_seen<=1)
    ok &= (state.family!=0)|(state.main_seen==0)
    ok &= (state.main_seen!=0)|jnp.all(state.main_proof==0,axis=-1)
    return ok&jnp.all(visible_ok,axis=(1,2))


def open_frame(state,code,valid):
    ok=(state.family==0)&(state.main_seen==0)&(code>=NOOP)&(code<=TRANSFORM)
    use=valid&ok
    return state._replace(family=jnp.where(use,code,state.family)),ok|~valid


def main(state,code,proof,valid):
    ok=(state.family==code)&(state.main_seen==0)
    return state._replace(main_seen=jnp.where(valid&ok,1,state.main_seen),
        main_proof=jnp.where((valid&ok)[:,None],proof,state.main_proof)),ok|~valid


def checkpoint(state,beliefs,source_tables,mask):
    cache,beliefs,ok=snapshot.refresh(state.controller.visible,beliefs,source_tables,source_refresh=mask)
    return state._replace(controller=state.controller._replace(visible=cache)),beliefs,jnp.all(ok,axis=(1,2))


def close_frame(state,beliefs,source_tables,control,control_side,valid):
    _,basis_ok=snapshot.read(state.controller.visible,beliefs,source_tables)
    ok=(state.family>=NOOP)&(state.family<=TRANSFORM)
    ok &= state.main_seen==jnp.where(state.family==NOOP,0,1)
    control_kind=control[:,0]
    arrival=(control_kind>=tendency.SWITCH)&(control_kind<=tendency.REPLACE)
    callback_main=arrival|(control_kind==tendency.MOVE)|(control_kind==tendency.TURN)
    unsupported=(control_kind==tendency.FAINT)|(control_kind==tendency.CANT)|(control_kind==tendency.UPKEEP)
    expected=jnp.where(state.family==ARRIVAL,arrival,jnp.where(state.family==MOVE,control_kind==tendency.MOVE,
        jnp.where(state.family==TURN,control_kind==tendency.TURN,
        jnp.where(state.family==NOOP,~callback_main&~unsupported,control_kind==tendency.OTHER))))
    ok &= expected
    proof=jnp.stack((control_side,control_kind,jnp.where(arrival,control[:,1],control[:,2]),
        control[:,3],control[:,5],control[:,4]),axis=-1)
    ok &= (state.family==NOOP)|jnp.all(proof==state.main_proof,axis=-1)
    ok &= jnp.all(basis_ok,axis=(1,2))
    use=valid&ok
    return state._replace(family=jnp.where(use,0,state.family),main_seen=jnp.where(use,0,state.main_seen),
        main_proof=jnp.where(use[:,None],0,state.main_proof)),ok|~valid


def commit(ok,before,after):
    """Commit batched state while retaining the cache's global namespace words."""
    choose=lambda a,b:refresh._choose(ok,a,b)
    controller=before.controller._replace(
        binding=choose(before.controller.binding,after.controller.binding),
        visible=refresh._cache_commit(ok,before.controller.visible,after.controller.visible),
        first_selected=choose(before.controller.first_selected,after.controller.first_selected),
        choice_excluded=choose(before.controller.choice_excluded,after.controller.choice_excluded))
    return StreamState(controller,choose(before.family,after.family),choose(before.main_seen,after.main_seen),
        choose(before.main_proof,after.main_proof))
