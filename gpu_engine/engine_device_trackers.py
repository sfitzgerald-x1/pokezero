"""Small public-knowledge device carry for direct engine observations.

No event tape, source callback proof, host transfer or observation admission is
involved. This first cut observes only active endpoints of an actual engine
step. Transient within-step installations and hook-only facts are explicitly
deferred; this module does not establish parity or throughput qualification.
"""
from typing import NamedTuple
import jax
import jax.numpy as j

SCHEMA='pokezero.engine_device_public_trackers.v1'
BAD_ACTIVE=1
BAD_PUBLIC_VALUE=2
BEFORE_ACTIVE_MISMATCH=4
BAD_OBSERVER=8
BAD_ROOT=16
CAPACITY=6
DEFERRED={
    'opponent_ability_item':'Zero until an actual public ability/item reveal hook.',
    'opponent_live_types':'Zero; encoder may use public species static base types with its own manifest scope.',
    'observed_moves_pp_lastmoves_tendencies':'Zero/deferred until actual called-move and public outcome hooks.',
    'belief_posterior':'None; no sampled hidden state or fabricated narrowing.',
    'transient_installations':'Before/after endpoints do not expose multiple intermediate installations in one step.',
    'own_order':'Active-first observed net-switch swaps only; per-install hook needed for exact transient request order.',
    'bench_public_changes':'Hidden/previously seen bench fields are not polled; party-wide public cure hooks are deferred.',
    'public_weather_remaining':'Not exposed; true engine duration is not a public timer receipt.',
}

class PublicView(NamedTuple):
    own_order:object
    species:object
    level:object
    gender:object
    hp_fraction:object
    status:object
    active:object
    present:object
    fainted:object
    types:object
    boosts:object
    ability:object
    item:object
    error:object
    posterior_weights:object=None
    turn:object=None
    weather:object=None
    spikes:object=None

class Trackers(NamedTuple):
    physical_to_public:object # [B,2,6], private identity cursor only; never an encoder feature
    active_physical:object    # [B,2]
    active_slot:object        # [B,2]
    next_slot:object          # [B,2]
    own_order:object          # [B,2,6], observed net-install permutation
    species:object           # remaining per-member arrays use PUBLIC encounter slots
    level:object
    gender:object
    hp_percent:object
    status:object
    boosts:object
    present:object
    fainted:object
    turn:object
    weather:object
    spikes:object
    error:object             # diagnostic input-contract bits, no engine/capture gate
    observed_moves:object=None
    pp_used:object=None
    last_moves:object=None
    tendencies:object=None
    belief_mask:object=None


def _shape(x,shape,dtype,name):
    if getattr(x,'shape',None)!=shape or getattr(x,'dtype',None)!=j.dtype(dtype):
        raise ValueError('device public tracker shape/dtype: '+name)


def _state_shape(state):
    b=state.active.shape[0]
    _shape(state.active,(b,2),j.int32,'active')
    for name in ('species','level','gender','hp','max_hp','status'):
        _shape(getattr(state,name),(b,2,6),j.int32,name)
    _shape(state.boosts,(b,2,6,7),j.int32,'boosts')
    for name in ('turn','weather'):_shape(getattr(state,name),(b,),j.int32,name)
    _shape(state.spikes,(b,2),j.int32,'spikes')
    return b


def _percent(hp,maximum):
    positive=j.maximum(maximum,1);hp=j.clip(hp,0,positive)
    shown=(100*hp+positive-1)//positive
    return j.where((hp<positive)&(shown==100),99,shown)


def _observe(t,state,side,physical,rows_active,*,identity):
    """Observe only a known active endpoint; never poll an opponent bench."""
    b=len(rows_active);r=j.arange(b);p=j.clip(physical,0,5)
    slot=t.physical_to_public[r,side,p];at=j.clip(slot,0,5)
    use=rows_active&(physical>=0)&(physical<6)&(slot>=0)&(slot<6)
    hp=state.hp[r,side,p];maximum=state.max_hp[r,side,p]
    valid=(maximum>0)&(hp>=0)&(hp<=maximum)&(state.status[r,side,p]>=0)&(state.status[r,side,p]<=6)
    if identity:
        valid &= (state.species[r,side,p]>0)&(state.level[r,side,p]>0)&(state.level[r,side,p]<=100)&(state.gender[r,side,p]>=-1)&(state.gender[r,side,p]<=2)
    error=t.error|j.where(use&~valid,BAD_PUBLIC_VALUE,0)
    values={}
    for name,value in [('hp_percent',_percent(hp,maximum)),('status',state.status[r,side,p]),('fainted',hp==0)]:
        old=getattr(t,name);values[name]=old.at[r,side,at].set(j.where(use,value,old[r,side,at]))
    old=t.boosts;values['boosts']=old.at[r,side,at].set(j.where(use[:,None],state.boosts[r,side,p],old[r,side,at]))
    if identity:
        for name in ('species','level','gender'):
            old=getattr(t,name);values[name]=old.at[r,side,at].set(j.where(use,getattr(state,name)[r,side,p],old[r,side,at]))
        old=t.present;values['present']=old.at[r,side,at].set(j.where(use,state.species[r,side,p]>0,old[r,side,at]))
    return t._replace(error=error,**values)


def _install(t,state,side,active_rows):
    b=len(active_rows);r=j.arange(b);physical=state.active[:,side];p=j.clip(physical,0,5)
    old=t.physical_to_public[r,side,p];fresh=old<0;slot=j.where(fresh,t.next_slot[:,side],old)
    good=(physical>=0)&(physical<6)&(slot>=0)&(slot<6)
    use=active_rows&good;at=j.clip(slot,0,5)
    mapping=t.physical_to_public.at[r,side,p].set(j.where(use,at,old))
    next_slot=t.next_slot.at[:,side].set(t.next_slot[:,side]+(use&fresh).astype(j.int32))
    current=t.active_physical.at[:,side].set(j.where(use,physical,t.active_physical[:,side]))
    current_slot=t.active_slot.at[:,side].set(j.where(use,at,t.active_slot[:,side]))
    t=t._replace(physical_to_public=mapping,next_slot=next_slot,active_physical=current,active_slot=current_slot,
        error=t.error|j.where(active_rows&~good,BAD_ACTIVE,0))
    return _observe(t,state,side,physical,use,identity=True)


def initialize(state):
    """Only current initial actives are publicly revealed; own data stays in state."""
    b=_state_shape(state);z=j.zeros((b,2,6),j.int32);no=j.zeros((b,2,6),bool)
    order=j.broadcast_to(j.arange(6,dtype=j.int32),(b,2,6))
    t=Trackers(j.full((b,2,6),-1,j.int32),j.full((b,2),-1,j.int32),j.full((b,2),-1,j.int32),j.zeros((b,2),j.int32),order,
        z,z,j.full_like(z,-1),z,z,j.zeros((b,2,6,7),j.int32),no,no,state.turn,state.weather,state.spikes,j.zeros(b,j.int32))
    r=j.arange(b);yes=j.ones(b,bool)
    for side in range(2):
        physical=j.clip(state.active[:,side],0,5)
        side_order=t.own_order[:,side].at[r,physical].set(0).at[:,0].set(physical)
        t=t._replace(own_order=t.own_order.at[:,side].set(side_order))
        t=_install(t,state,side,yes)
    return t


def update(trackers,before,after,actions,*,active_rows=None):
    """Pair with one actual bare engine step, inside the caller's same JIT.

    Actions are retained as an API operand but are not interpreted as public
    move declarations: called moves/PP require later actual execution hooks.
    Inactive rows are unchanged, including all public knowledge and errors.
    """
    b=_state_shape(before)
    if _state_shape(after)!=b:raise ValueError('same device tracker batch required')
    _shape(actions,(b,2),j.int32,'actions')
    active_rows=j.ones(b,bool)if active_rows is None else active_rows
    _shape(active_rows,(b,),j.bool_,'active_rows')
    same=j.all(before.active==trackers.active_physical,axis=1)
    t=trackers._replace(error=trackers.error|j.where(active_rows&~same,BEFORE_ACTIVE_MISMATCH,0))
    use=active_rows&same;r=j.arange(b)
    for side in range(2):
        # The outgoing endpoint was public before this actual step. Updating
        # it does not read any unrelated hidden/previously-seen bench member.
        t=_observe(t,after,side,before.active[:,side],use,identity=False)
        incoming=after.active[:,side];p=j.clip(incoming,0,5)
        old_order=t.own_order[:,side];position=j.argmax(old_order==p[:,None],axis=1)
        swapped=old_order.at[r,position].set(old_order[:,0]).at[:,0].set(p)
        changed=use&(incoming>=0)&(incoming<6)&(incoming!=before.active[:,side])
        t=t._replace(own_order=t.own_order.at[:,side].set(j.where(changed[:,None],swapped,old_order)))
        t=_install(t,after,side,use)
    return t._replace(turn=j.where(use,after.turn,t.turn),weather=j.where(use,after.weather,t.weather),spikes=j.where(use[:,None],after.spikes,t.spikes))


def readout(trackers,observer_sides):
    """No engine/world argument: opposing private data cannot enter this view."""
    b=len(trackers.error);_shape(observer_sides,(b,),j.int32,'observer_sides')
    r=j.arange(b);owner=j.clip(observer_sides,0,1);foe=1-owner
    take=lambda name:getattr(trackers,name)[r,foe]
    present=take('present');active=(j.arange(6)[None,:]==trackers.active_slot[r,foe,None])&present
    z=j.zeros((b,6),j.int32)
    return PublicView(trackers.own_order[r,owner],take('species'),take('level'),take('gender'),
        take('hp_percent').astype(j.float32)/100.0,take('status'),active,present,take('fainted'),
        j.zeros((b,6,2),j.int32),take('boosts'),z,z,
        trackers.error|j.where((observer_sides<0)|(observer_sides>1),BAD_OBSERVER,0),None,
        trackers.turn,trackers.weather,trackers.spikes)


def clone_for_worlds(trackers,root_rows):
    """Clone root PUBLIC knowledge for flattened D leaves, never sampled state."""
    if getattr(root_rows,'ndim',None)!=1 or getattr(root_rows,'dtype',None)!=j.dtype(j.int32):
        raise ValueError('flat int32 root row indices required')
    valid=(root_rows>=0)&(root_rows<len(trackers.error));at=j.clip(root_rows,0,len(trackers.error)-1)
    def gather(value):
        result=value[at];mask=valid.reshape((len(valid),)+(1,)*(result.ndim-1))
        return j.where(mask,result,j.zeros_like(result))
    result=jax.tree.map(gather,trackers)
    return result._replace(error=j.where(valid,result.error,BAD_ROOT))
