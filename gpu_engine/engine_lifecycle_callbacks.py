"""Partial causal lifecycle receipts from the real engine, not state diffs.

Initial source reveals are side0,side1 BEFORE the SAME legacy initialize call.
Actual swaps are captured only at the transition's installed-active write,
before hazards. Source upkeep follows residual handlers; nextTurn precedes the
next ordinary request, including after a post-residual forced replacement.

All tapes/physical maps/details are INTERNAL. No complete SWITCH/raw-line or
173-column admission is claimed. Unknown gender/details remain unknown. A
Baton install advances private order but leaves public active unestablished
until its missing context is supported. Uninstrumented drag is a refusal, never
an installation inferred from the final state.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

from . import engine_public_events as public
from . import engine_request_order as request_order

SCHEMA='pokezero.engine_lifecycle_callbacks.v1'
TURN,UPKEEP,SWITCH_INSTALLED,ACTION_END,INITIAL_REVEAL=range(4,9)
UNKNOWN_DETAILS,UNKNOWN_BATON,MISSING_INSTALL,CLOCK=8,16,32,64


class Cursor(NamedTuple):
    """Admission covers lifecycle identity/clocks only, never encoder coverage."""
    stream_id: object
    engine_binding: object
    generation: object
    turn: object
    phase: object
    initialized: object
    admitted: object
    physical_active: object
    public_slot: object
    encounter: object
    established: object
    revealed_species_id: object
    physical_to_public: object
    next_slot: object


class PublicArrivalSnapshot(NamedTuple):
    stream_id: object
    engine_binding: object
    generation: object
    physical_active: object
    public_slot: object
    encounter: object
    established: object
    revealed_species_id: object


class LifecycleTape(NamedTuple):
    # The first eight fields retain the frozen CallbackTape's field ABI.
    words: object
    count: object
    refusal: object
    turn: object
    engine_binding: object
    source_binding: object
    move_binding: object
    producer_binding: object
    stream_id: object
    event_turn: object
    event_phase: object
    event_generation: object
    switch_details: object
    cursor: Cursor
    execution: object = None  # PRIVATE partial source-site accounting


class LeftoversContext(NamedTuple):
    """PRIVATE checked helper data, not public/source-order admission."""
    data: object


class StatusResidualContext(NamedTuple):
    """PRIVATE checked literal residual helper, never public/order credit."""
    data: object


class LifecycleResult(NamedTuple):
    state: object
    cursor: Cursor
    callbacks: LifecycleTape
    before_registry: PublicArrivalSnapshot
    after_registry: PublicArrivalSnapshot
    invalid_actions: object
    placeholder_moves: object
    weather_profile: object = None  # INTERNAL owning carry, never a public fact

    @property
    def full_line_consumers_complete(self):return False

    @property
    def full_encoder_admitted(self):return False

    @property
    def B4_qualified(self):return False


def _typed(value,shape,dtype):
    if (not isinstance(value,(jax.Array,jax.core.Tracer)) or value.shape!=shape or value.dtype!=jnp.dtype(dtype)):
        raise ValueError('exact immutable device lifecycle shape/dtype required')


def _valid(cursor):
    if type(cursor) is not Cursor or getattr(cursor.stream_id,'ndim',None)!=1:
        raise ValueError('registered lifecycle cursor required')
    b=cursor.stream_id.shape[0]
    for name in ('stream_id','turn','phase'):_typed(getattr(cursor,name),(b,),jnp.int32)
    for name in ('initialized','admitted'):_typed(getattr(cursor,name),(b,),jnp.bool_)
    _typed(cursor.engine_binding,(b,8),jnp.uint32)
    for name in ('generation','physical_active','public_slot','encounter','revealed_species_id','next_slot'):
        _typed(getattr(cursor,name),(b,2),jnp.int32)
    _typed(cursor.established,(b,2),jnp.bool_)
    _typed(cursor.physical_to_public,(b,2,6),jnp.int32)
    mapping=cursor.physical_to_public
    expected=jnp.where(jnp.arange(6)<cursor.next_slot[:,:,None],jnp.arange(6),6)
    mapped=jnp.all(jnp.sort(jnp.where(mapping>=0,mapping,6),axis=2)==expected,axis=(1,2))
    rows=jnp.arange(b)[:,None];sides=jnp.arange(2)[None,:]
    active=jnp.clip(cursor.physical_active,0,5)
    established_shape=jnp.all((cursor.physical_active>=0)&(cursor.physical_active<6)
        &(cursor.public_slot>=0)&(cursor.public_slot<6)
        &(mapping[rows,sides,active]==cursor.public_slot)&(cursor.encounter==cursor.public_slot+1),axis=1)
    new_shape=jnp.all(cursor.physical_active==-1,axis=1)&jnp.all(cursor.public_slot==-1,axis=1)
    new_shape &= jnp.all(cursor.encounter==0,axis=1)&jnp.all(~cursor.established,axis=1)
    new_shape &= jnp.all(cursor.next_slot==0,axis=1)&jnp.all(cursor.generation==0,axis=1)
    return (cursor.admitted & (cursor.stream_id>0) & (cursor.turn>=0)&mapped
        &jnp.all((mapping>=-1)&(mapping<6),axis=(1,2))
        &jnp.where(cursor.initialized,established_shape&(cursor.turn>0),new_shape)
        & ((cursor.phase==0)|(cursor.phase==1)) & jnp.all(cursor.generation>=0,axis=1)
        & jnp.all((cursor.next_slot>=0)&(cursor.next_slot<=6),axis=1))


def register(stream_ids,*,catalog,expected_binding,expected_engine_binding):
    """Constructor registration only: no public party/encounter assignment."""
    public.check_catalog(catalog,catalog.engine_table,expected_binding=expected_binding,
        expected_engine_binding=expected_engine_binding)
    if getattr(stream_ids,'ndim',None)!=1 or not stream_ids.shape[0]:
        raise ValueError('nonempty registered streams required')
    b=stream_ids.shape[0];_typed(stream_ids,(b,),jnp.int32)
    values=np.asarray(stream_ids)
    if np.any(values<=0) or len(set(values.tolist()))!=b:
        raise ValueError('unique positive constructor stream IDs required')
    z=jnp.zeros((b,2),jnp.int32);blank=jnp.full((b,2),-1,jnp.int32)
    return Cursor(stream_ids,jnp.broadcast_to(jnp.array(catalog.engine_binding,jnp.uint32),(b,8)),z,
        jnp.zeros(b,jnp.int32),jnp.zeros(b,jnp.int32),jnp.zeros(b,bool),jnp.ones(b,bool),
        blank,blank,z,jnp.zeros((b,2),bool),z,jnp.full((b,2,6),-1,jnp.int32),z)


def snapshot(cursor):
    ok=_valid(cursor)
    visible=ok[:,None]&cursor.established
    return PublicArrivalSnapshot(cursor.stream_id,cursor.engine_binding,cursor.generation,
        jnp.where(ok[:,None],cursor.physical_active,-1),jnp.where(visible,cursor.public_slot,-1),
        jnp.where(visible,cursor.encounter,0),visible,jnp.where(visible,cursor.revealed_species_id,0))


def _tape(cursor,catalog,capacity):
    b=cursor.stream_id.shape[0]
    base=public.initialize(b,capacity,catalog=catalog)
    base=base._replace(refusal=jnp.full_like(base.refusal,public.UNINSTRUMENTED))
    return LifecycleTape(*base[:3],cursor.turn,*base[4:],cursor.stream_id,
        jnp.zeros((b,capacity),jnp.int32),jnp.zeros((b,capacity),jnp.int32),
        jnp.zeros((b,capacity,2),jnp.int32),jnp.zeros((b,capacity,6),jnp.int32),cursor)


def stamp(tape,index,eligible):
    """Called by the common append primitive; old CallbackTape is unchanged."""
    if type(tape) is not LifecycleTape:raise ValueError('exact lifecycle callback tape required')
    rows=jnp.arange(tape.count.shape[0])
    def write(array,value):
        old=array[rows,index];mask=eligible.reshape((-1,)+(1,)*(old.ndim-1))
        return array.at[rows,index].set(jnp.where(mask,value,old))
    return tape._replace(event_turn=write(tape.event_turn,tape.cursor.turn),
        event_phase=write(tape.event_phase,tape.cursor.phase),
        event_generation=write(tape.event_generation,tape.cursor.generation))


def _clock(tape,kind,eligible):
    absent=jnp.full_like(tape.count,-1);z=jnp.zeros_like(tape.count)
    return public._append(tape,kind,absent,absent,absent,absent,(z,z,z,z,z),eligible)


def _turn(tape,eligible):
    eligible=eligible&tape.cursor.admitted
    safe=tape.cursor.turn<2**31-1
    ok=eligible&safe
    cursor=tape.cursor._replace(turn=tape.cursor.turn+ok.astype(jnp.int32),
        phase=jnp.where(ok,0,tape.cursor.phase),admitted=tape.cursor.admitted&(~eligible|safe))
    tape=tape._replace(cursor=cursor,turn=cursor.turn,refusal=tape.refusal|jnp.where(eligible&~safe,CLOCK,0))
    return _clock(tape,TURN,ok)


def _reveal(tape,state,side,outgoing,incoming,*,eligible,replacement,baton,initial):
    rows=jnp.arange(tape.count.shape[0]);cursor=tape.cursor
    eligible=eligible&cursor.admitted
    requested=eligible
    slot=jnp.clip(incoming,0,5)
    old=cursor.physical_to_public[rows,side,slot]
    new=old<0
    public_slot=jnp.where(new,cursor.next_slot[rows,side],old)
    safe=(incoming>=0)&(incoming<6)&(public_slot>=0)&(public_slot<6)
    safe &= initial|(cursor.generation[rows,side]<2**31-1)
    eligible &= safe
    details=jnp.stack([getattr(state,name)[rows,side,slot] for name in
        ('species','level','gender','hp','max_hp','status')],axis=1)
    species,level,gender,hp,maximum,status=details.T
    known=(species>0)&(level>0)&(level<=100)&(gender>=0)&(gender<=2)&(hp>0)&(hp<=maximum)&(status>=0)&(status<=6)
    before=tape.count;index=jnp.minimum(before,tape.words.shape[1]-1)
    room=before<tape.words.shape[1]
    flags=jnp.asarray(baton,jnp.int32);replacing=jnp.asarray(replacement,jnp.int32)
    absent=jnp.full_like(before,-1);z=jnp.zeros_like(before)
    tape=public._append(tape,INITIAL_REVEAL if initial else SWITCH_INSTALLED,
        absent if initial else side,absent if initial else outgoing,side,incoming,(replacing,flags,z,z,z),eligible)
    old_details=tape.switch_details[rows,index]
    tape=tape._replace(switch_details=tape.switch_details.at[rows,index].set(
        jnp.where((eligible&room)[:,None],details,old_details)))
    mapping=cursor.physical_to_public.at[rows,side,slot].set(jnp.where(eligible,public_slot,old))
    def active(name,value):
        previous=getattr(cursor,name)
        return previous.at[rows,side].set(jnp.where(eligible,value,previous[rows,side]))
    gen=cursor.generation.at[rows,side].add((eligible&(not initial)).astype(jnp.int32))
    # Baton is a genuine new reveal/private swap, but its unported public flag
    # context must never silently reuse an old public encounter as established.
    established=active('established',~jnp.asarray(baton))
    cursor=cursor._replace(generation=gen,physical_to_public=mapping,
        next_slot=active('next_slot',cursor.next_slot[rows,side]+(eligible&new).astype(jnp.int32)),
        physical_active=active('physical_active',incoming),public_slot=active('public_slot',public_slot),
        encounter=active('encounter',public_slot+1),established=established,
        revealed_species_id=active('revealed_species_id',species),
        admitted=cursor.admitted&~(eligible&~room)&~(requested&~safe))
    return tape._replace(cursor=cursor,refusal=tape.refusal
        |jnp.where(eligible&~known,UNKNOWN_DETAILS,0)|jnp.where(eligible&baton,UNKNOWN_BATON,0)
        |jnp.where(requested&~safe,CLOCK,0))


def initialize_with_lifecycle(state,table,*,cursor,catalog,expected_binding,expected_engine_binding,
                              tie_coin=None,capacity=64,execution_compiled=None):
    """Own NEW -> actual reveals0,1 -> SAME legacy initialize -> TURN1.

    Constructor wrappers must pass the uninitialized state_from_pool(table=None)
    result and any proven constructor gender values, not a post-step snapshot.
    """
    if execution_compiled is not None and isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
        raise ValueError('checked accounting constructor must remain outside outer JIT')
    from . import transition
    public.check_catalog(catalog,table,expected_binding=expected_binding,expected_engine_binding=expected_engine_binding)
    execution_tables=None
    if execution_compiled is not None:
        from . import engine_execution_completeness as execution
        execution_tables,_,adapter=execution._checked(execution_compiled)
        if adapter.causal.projector.catalog is not catalog:
            raise ValueError('same original lifecycle catalog required for accounting constructor')
    ok=_valid(cursor)&~cursor.initialized&(cursor.turn==0)&(cursor.phase==0)&~state.done&(state.turn==0)
    ok &= jnp.all(cursor.engine_binding==jnp.array(catalog.engine_binding,jnp.uint32),axis=1)
    ok &= (jnp.all(cursor.next_slot==0,axis=1)&jnp.all(cursor.physical_to_public==-1,axis=(1,2))
        &jnp.all(cursor.generation==0,axis=1)&jnp.all(cursor.physical_active==-1,axis=1)
        &jnp.all(~cursor.established,axis=1))
    ok &= jnp.all(state.entry_pending,axis=1)&jnp.all(state.active==0,axis=1)
    ok &= jnp.all(state.hp[jnp.arange(state.hp.shape[0])[:,None],jnp.arange(2),state.active]>0,axis=1)
    ok &= table.entry is not None
    before=snapshot(cursor)
    cursor=cursor._replace(admitted=ok)
    tape=_tape(cursor,catalog,capacity)._replace(refusal=jnp.full_like(cursor.turn,public.UNINSTRUMENTED)
        |jnp.where(~ok,CLOCK,0))
    if execution_tables is not None:
        ledger=execution.initialize(cursor.stream_id,compiled=execution_compiled,capacity=32,receipt_capacity=capacity)
        tape=execution.begin_on_tape(tape,ledger,execution_tables,active=ok)
    for side in (0,1):
        seat=jnp.full_like(cursor.turn,side)
        tape=_reveal(tape,state,seat,jnp.full_like(cursor.turn,-1),state.active[:,side],
            eligible=ok,replacement=jnp.zeros_like(ok),baton=jnp.zeros_like(ok),initial=True)
    # Missing Ability Start/Lum cure attribution is refused before callbacks;
    # static tables prove no cure possible for other known-status arrivals.
    from . import engine_status_callbacks as status_callbacks
    tape=status_callbacks.initial_context(tape,state,table,ok)
    updated,tape=transition.initialize(state,table,eligible=ok,tie_coin=tie_coin,_weather_trace=tape)
    complete=ok&jnp.all(~updated.entry_pending,axis=1)&tape.cursor.admitted
    tape=tape._replace(cursor=tape.cursor._replace(initialized=complete,admitted=complete))
    tape=_turn(tape,complete)
    return LifecycleResult(updated,tape.cursor,tape,before,snapshot(tape.cursor),jnp.zeros_like(ok),jnp.zeros_like(cursor.turn))


def begin(tape,state,live,invalid):
    ok=_valid(tape.cursor)&tape.cursor.initialized
    ok &= jnp.all(tape.cursor.engine_binding==tape.engine_binding,axis=1)
    hp=state.hp[jnp.arange(state.hp.shape[0])[:,None],jnp.arange(2),state.active]
    replacing=jnp.any(state.forced_switch|(hp<=0),axis=1)
    expected=state.turn+(~replacing).astype(jnp.int32)
    ok &= (tape.cursor.turn==expected)&jnp.all(tape.cursor.physical_active==state.active,axis=1)
    ok &= jnp.all(~state.entry_pending,axis=1)
    ok &= jnp.where(replacing,jnp.where(state.pending_residual,tape.cursor.phase==0,tape.cursor.phase==1),tape.cursor.phase==0)
    return tape._replace(cursor=tape.cursor._replace(admitted=ok&~invalid),turn=tape.cursor.turn,
        refusal=tape.refusal|jnp.where(~ok,CLOCK,0))


def installed(tape,state,side,outgoing,incoming,*,eligible,replacement,baton):
    return _reveal(tape,state,side,outgoing,incoming,eligible=eligible,replacement=replacement,baton=baton,initial=False)


def action_end(tape,eligible):
    return _clock(tape,ACTION_END,eligible)


def finish(tape,state,*,live,replacement,pending_before,residual_ready,defer_residual,turn_context):
    # At the SAME residual completion boundary after Truant/charge callbacks.
    upkeep=residual_ready&~state.done&tape.cursor.admitted
    cursor=tape.cursor._replace(phase=jnp.where(upkeep,1,tape.cursor.phase))
    tape=_clock(tape._replace(cursor=cursor),UPKEEP,upkeep)
    next_turn=live&~state.done&~jnp.any(state.forced_switch,axis=1)&(
        (turn_context&~defer_residual)|(replacement&~pending_before))
    tape=_turn(tape,next_turn)
    # Validate missing producers; NEVER turn an unrecorded active diff (drag)
    # into a switch receipt or an inferred public identity.
    matches=jnp.all(tape.cursor.physical_active==state.active,axis=1)
    ok=tape.cursor.admitted&(~live|matches)&((tape.refusal&public.OVERFLOW)==0)
    return tape._replace(cursor=tape.cursor._replace(admitted=ok),turn=tape.cursor.turn,
        refusal=tape.refusal|jnp.where(live&~matches,MISSING_INSTALL,0))


def trace_step(state,actions,table,*,cursor,catalog,expected_binding,expected_engine_binding,
               chance_inputs=None,capacity=64,leftovers_compiled=None,status_residual_compiled=None):
    # A checked host boundary cannot be captured even when only cursor/chance
    # (or no meaningful argument) is dynamic. Inspect this FRESH scalar, not
    # selected input leaves, before any host guard or cached core invocation.
    if (leftovers_compiled is not None or status_residual_compiled is not None) and isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
        raise ValueError('checked lifecycle helper API must remain outside outer JIT')
    from . import transition
    public.check_catalog(catalog,table,expected_binding=expected_binding,expected_engine_binding=expected_engine_binding)
    leftovers_context=None
    if leftovers_compiled is not None:
        from . import engine_leftovers_callbacks as leftovers_callbacks
        data=leftovers_callbacks.check_compiled(leftovers_compiled)
        if leftovers_callbacks._COMPILED[leftovers_compiled][4] is not catalog:
            raise ValueError('same original engine catalog required for Leftovers caller')
        leftovers_context=LeftoversContext(data)
    status_residual_context=None
    if status_residual_compiled is not None:
        from . import engine_status_residual_callbacks as status_residual_callbacks
        data=status_residual_callbacks.check_compiled(status_residual_compiled)
        if status_residual_callbacks._COMPILED[status_residual_compiled][4] is not catalog:
            raise ValueError('same original engine catalog required for status residual caller')
        status_residual_context=StatusResidualContext(data)
    before=snapshot(cursor);tape=_tape(cursor,catalog,capacity)
    chance_inputs=transition.sample_chance_inputs(state) if chance_inputs is None else chance_inputs
    result=transition.step_with_chance(state,actions,table,chance_inputs,_public_trace=tape,_lifecycle_trace=True,
        _leftovers_context=leftovers_context,_status_residual_context=status_residual_context)
    tape=result.callbacks
    return LifecycleResult(result.step.state,tape.cursor,tape,before,snapshot(tape.cursor),
        result.step.invalid_actions,result.step.placeholder_moves)


def apply_installations(order,frame):
    """Feed actual SWAP receipts to the merged owner permutation in order."""
    if type(frame) is not LifecycleResult:raise ValueError('actual lifecycle result required')
    tape=frame.callbacks;b=tape.count.shape[0];rows=jnp.arange(b)
    for index in range(tape.words.shape[1]):
        raw=tape.words[:,index];active=index<tape.count
        generation=jnp.where(active,tape.event_generation[:,index][rows,order.owner_side],order.generation)
        receipt=request_order.SwitchInstalled(tape.stream_id,order.owner_side,
            jnp.broadcast_to(tape.engine_binding,(b,8)),generation,
            active&(raw[:,0]==SWITCH_INSTALLED)&(raw[:,1]==order.owner_side),raw[:,2],raw[:,4])
        order=request_order.install_switch(order,receipt)
    valid=frame.cursor.admitted&((tape.refusal&public.OVERFLOW)==0)&(
        order.generation==frame.cursor.generation[rows,order.owner_side])
    return order._replace(admitted=order.admitted&valid,
        request_to_physical=jnp.where(valid[:,None],order.request_to_physical,-1))
