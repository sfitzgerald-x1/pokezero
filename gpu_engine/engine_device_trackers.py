"""Small public-knowledge device carry for direct engine observations.

No event tape, source callback proof, host transfer or observation admission is
involved. The legacy initialize/update API observes active endpoints. The opt-in
transition.initialize_with_public/step_with_public path fills actual write
hooks and appended knowledge fields; this module does not establish parity or throughput qualification.
"""
from typing import NamedTuple
import jax
import jax.numpy as j
from . import lane_index as _li

SCHEMA='pokezero.engine_device_public_trackers.v3'
BAD_ACTIVE=1
BAD_PUBLIC_VALUE=2
BEFORE_ACTIVE_MISMATCH=4
BAD_OBSERVER=8
BAD_ROOT=16
CAPACITY=6
DEFERRED={
    'opponent_ability_item':'Zero until an actual public ability/item reveal hook.',
    'opponent_live_types':'Zero; encoder may use public species static base types with its own manifest scope.',
    'observed_moves_pp_lastmoves_tendencies':'Legacy endpoint API only; full hook path supplies actual declarations and marginal tendencies.',
    'belief_posterior':'Tracker retains hard source masks; encoder owns the effective fallback/posterior projector.',
    'transient_installations':'Legacy endpoint API only; public-step hooks observe each actual installation.',
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
    belief_mask:object=None
    revealed_moves:object=None
    move_pp_used:object=None
    last_moves:object=None
    borrowed_ability:object=None
    own_last_moves:object=None
    mon_features:object=None
    own_mon_features:object=None
    field_features:object=None
    own_borrowed_ability:object=None
    volatiles:object=None
    own_volatiles:object=None
    item_removed:object=None
    own_item_removed:object=None
    stats_features:object=None
    belief_pin:object=None
    belief_pin_active:object=None

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
    shown_ability:object=None
    shown_item:object=None
    shown_types:object=None
    borrowed_ability:object=None
    mon_features:object=None
    field_features:object=None
    original_ability:object=None
    damage_exact_dealt:object=None
    damage_exact_taken:object=None
    public_volatiles:object=None
    item_removed:object=None
    tendency:object=None
    damage_history:object=None    # Current public-round sums + owning HP baseline
    negative_items:object=None
    visible_belief:object=None     # standing pins + explicitly source-refreshed basis
    public_port:object=None
    # [B,2,6] permutation of PUBLIC encounter slots: Showdown's request-order
    # rule (each install swaps the installed member with position 0) replayed
    # from public installs only. Reset only by initialize (identity: the lead
    # holds public slot 0); never encoded, read out or redacted.
    public_order:object=None


class Tendency(NamedTuple):
    moved:object                  # [B,2], first move of current stay
    opportunity_turn:object       # [B,2], public turn already counted
    switches:object               # [B,2], voluntary original switches
    opportunities:object          # [B,2], ordinary move/cant/switch decisions
    matchup:object                # [B,2,6,6,2], public encounter switch/stay counts


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


def _gated(old,use,value):return j.where(use,value,old)


def _observe(t,state,side,physical,rows_active,*,identity):
    """Observe only a known active endpoint; never poll an opponent bench."""
    b=len(rows_active);r=j.arange(b);p=j.clip(physical,0,5)
    slot=_li.take(t.physical_to_public,r,side,p);at=j.clip(slot,0,5)
    use=rows_active&(physical>=0)&(physical<6)&(slot>=0)&(slot<6)
    hp=_li.take(state.hp,r,side,p);maximum=_li.take(state.max_hp,r,side,p)
    valid=(maximum>0)&(hp>=0)&(hp<=maximum)&(_li.take(state.status,r,side,p)>=0)&(_li.take(state.status,r,side,p)<=6)
    if identity:
        valid &= (_li.take(state.species,r,side,p)>0)&(_li.take(state.level,r,side,p)>0)&(_li.take(state.level,r,side,p)<=100)&(_li.take(state.gender,r,side,p)>=-1)&(_li.take(state.gender,r,side,p)<=2)
    error=t.error|j.where(use&~valid,BAD_PUBLIC_VALUE,0)
    values={}
    for name,value in [('hp_percent',_percent(hp,maximum)),('status',_li.take(state.status,r,side,p)),('fainted',hp==0)]:
        values[name]=_li.update(getattr(t,name),r,(side,at),_gated,use,value)
    values['boosts']=_li.update(t.boosts,r,(side,at),_gated,use[:,None],_li.take(state.boosts,r,side,p))
    if identity:
        for name in ('species','level','gender'):
            values[name]=_li.update(getattr(t,name),r,(side,at),_gated,use,_li.take(getattr(state,name),r,side,p))
        values['present']=_li.update(t.present,r,(side,at),_gated,use,_li.take(state.species,r,side,p)>0)
    return t._replace(error=error,**values)


def _install(t,state,side,active_rows):
    # side is either a scalar seat or one actual seat per row (speed order).
    # Pair EVERY seat gather/scatter with its row; [:, side] cross-products a
    # vector side into [B,B] and mixes independent games.
    b=len(active_rows);r=j.arange(b);physical=_li.take(state.active,r,side);p=j.clip(physical,0,5)
    old=_li.take(t.physical_to_public,r,side,p);fresh=old<0;slot=j.where(fresh,_li.take(t.next_slot,r,side),old)
    good=(physical>=0)&(physical<6)&(slot>=0)&(slot<6)
    use=active_rows&good;at=j.clip(slot,0,5)
    mapping=_li.put(t.physical_to_public,r,(side,p),j.where(use,at,old))
    next_slot=_li.put(t.next_slot,r,(side,),_li.take(t.next_slot,r,side)+(use&fresh).astype(j.int32))
    current=_li.put(t.active_physical,r,(side,),j.where(use,physical,_li.take(t.active_physical,r,side)))
    current_slot=_li.put(t.active_slot,r,(side,),j.where(use,at,_li.take(t.active_slot,r,side)))
    t=t._replace(physical_to_public=mapping,next_slot=next_slot,active_physical=current,active_slot=current_slot,
        error=t.error|j.where(active_rows&~good,BAD_ACTIVE,0))
    return _observe(t,state,side,physical,use,identity=True)


def public_order_install(t,side,use):
    """Request-order rule in public-slot coordinates, after an actual install.

    The member just installed (its public slot already assigned by _install)
    swaps places with position 0; a no-op when it is already first. Rows
    outside `use` and carries without the field are unchanged.
    """
    if t.public_order is None:return t
    r=j.arange(len(t.error));slot=_li.take(t.active_slot,r,side);order=_li.take(t.public_order,r,side)
    hit=order==slot[:,None];found=j.any(hit,axis=1);position=j.argmax(hit,axis=1)
    swapped=_li.put(order,r,(position,),order[:,0]).at[:,0].set(slot)
    keep=j.broadcast_to(j.asarray(use,bool),found.shape)&found
    return t._replace(public_order=_li.put(t.public_order,r,(side,),j.where(keep[:,None],swapped,order)))


def initialize(state):
    """Only current initial actives are publicly revealed; own data stays in state."""
    b=_state_shape(state);z=j.zeros((b,2,6),j.int32);no=j.zeros((b,2,6),bool)
    order=j.broadcast_to(j.arange(6,dtype=j.int32),(b,2,6))
    # public_order starts as identity: the lead's install below takes next_slot0.
    t=Trackers(j.full((b,2,6),-1,j.int32),j.full((b,2),-1,j.int32),j.full((b,2),-1,j.int32),j.zeros((b,2),j.int32),order,
        z,z,j.full_like(z,-1),z,z,j.zeros((b,2,6,7),j.int32),no,no,state.turn,state.weather,state.spikes,j.zeros(b,j.int32),
        public_order=order)
    r=j.arange(b);yes=j.ones(b,bool)
    for side in range(2):
        physical=j.clip(state.active[:,side],0,5)
        side_order=_li.put(t.own_order[:,side],r,(physical,),0).at[:,0].set(physical)
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
        swapped=_li.put(old_order,r,(position,),old_order[:,0]).at[:,0].set(p)
        changed=use&(incoming>=0)&(incoming<6)&(incoming!=before.active[:,side])
        t=t._replace(own_order=t.own_order.at[:,side].set(j.where(changed[:,None],swapped,old_order)))
        t=_install(t,after,side,use)
        # Same request-order rule as own_order, once per observed net install.
        t=public_order_install(t,side,changed)
    return t._replace(turn=j.where(use,after.turn,t.turn),weather=j.where(use,after.weather,t.weather),spikes=j.where(use[:,None],after.spikes,t.spikes))


def readout(trackers,observer_sides):
    """No engine/world argument: opposing private data cannot enter this view."""
    b=len(trackers.error);_shape(observer_sides,(b,),j.int32,'observer_sides')
    r=j.arange(b);owner=j.clip(observer_sides,0,1);foe=1-owner
    take=lambda name:_li.take(getattr(trackers,name),r,foe)
    present=take('present');active=(j.arange(6)[None,:]==trackers.active_slot[r,foe,None])&present
    z=j.zeros((b,6),j.int32)
    optional=lambda name:None if getattr(trackers,name) is None else take(name)
    mon=None if trackers.mon_features is None else {key:_li.take(value,r,foe) for key,value in trackers.mon_features.items()}
    stats=None
    if trackers.tendency is not None:
        td=trackers.tendency
        facing=j.clip(_li.take(trackers.active_slot,r,owner),0,5)
        pair=td.matchup[r[:,None],foe[:,None],j.arange(6)[None,:],facing[:,None]]
        pair=j.where((_li.take(trackers.active_slot,r,owner)>=0)[:,None,None],pair,0)
        mon={**mon,'switched_vs_active':pair[...,0],'stayed_vs_active':pair[...,1]}
        stats={'opp_switch_count':_li.take(td.switches,r,foe),
            'opp_decision_opportunities':_li.take(td.opportunities,r,foe),
            'my_switch_turns':_li.take(td.switches,r,owner)}
    own_mon=None
    own_last=None
    own_borrowed=None
    own_vol=None
    own_removed=None
    if trackers.mon_features is not None:
        physical=_li.take(trackers.own_order,r,owner)
        slots=trackers.physical_to_public[r[:,None],owner[:,None],physical]
        def own(value):
            got=value[r[:,None],owner[:,None],j.clip(slots,0,5)]
            return j.where(slots>=0,got,0)
        # Original tendency writer overlays only the opponent six tokens.
        own_mon={key:own(value) for key,value in trackers.mon_features.items()
            if key not in ('switched_before_attack','stayed_and_attacked','turns_active_total')}
        own_last=own(trackers.last_moves)
        own_borrowed=own(trackers.borrowed_ability)
        own_removed=own(trackers.item_removed)
        if trackers.public_volatiles is not None:
            got=trackers.public_volatiles[r[:,None],owner[:,None],j.clip(slots,0,5)]
            own_vol=j.where((slots>=0)[:,:,None],got,False)
        # Original observer-relative damage semantics: own victim HP is exact,
        # opposing victim HP comes from the shown-percent stream.
        mon['last_damage_dealt']=_li.take(trackers.damage_exact_dealt,r,foe)
        own_mon['last_damage_taken']=own(trackers.damage_exact_taken)
    fields=None
    if trackers.field_features is not None:
        raw=trackers.field_features
        fields={key:(_li.take(value,r,owner) if value.ndim==2 else value) for key,value in raw.items()}
        wish_turns=raw['wish_turns'];wish_pending=raw['wish_pending']
        if 'wish_set_turn' in raw:
            declared=raw['wish_set_turn'];elapsed=trackers.turn[:,None]-declared
            wish_pending=(declared>=0)&(elapsed<=1)
            wish_turns=j.where((declared>=0)&(elapsed>=0)&(elapsed<=1),2-elapsed,0)
        fields.update(self_wish_turns=_li.take(wish_turns,r,owner),opp_wish_turns=_li.take(wish_turns,r,foe),
            self_wish_pending=_li.take(wish_pending,r,owner),opp_wish_pending=_li.take(wish_pending,r,foe),
            weather_turns=j.where(trackers.weather>0,j.where(raw['weather_permanent'],5,j.maximum(0,5-raw['weather_upkeeps'])),0))
    visible=trackers.visible_belief
    belief_mask=optional('belief_mask') if visible is None else _li.take(visible.visible_hard,r,foe)
    if trackers.public_port is not None:
        h=trackers.public_port
        active_slot=_li.take(trackers.active_slot,r,foe)
        loaf=_li.take(h.known,r,foe)&_li.take(h.loaf,r,foe)
        mon={**({}if mon is None else mon),'truant_loaf':j.where(j.arange(6)[None,:]==active_slot[:,None],loaf[:,None],False)}
        own_active=_li.take(trackers.own_order,r,owner)==trackers.active_physical[r,owner,None]
        own_mon={**({}if own_mon is None else own_mon),'truant_loaf':own_active&(_li.take(h.known,r,owner)&_li.take(h.loaf,r,owner))[:,None]}
        hazard=h.hazards.sum(axis=-1)
        fields={**({}if fields is None else fields),
            'self_sleep_clause':_li.take(h.holder,r,owner)>=0,'opp_sleep_clause':_li.take(h.holder,r,foe)>=0,
            'sleep_clause_blocks_self':j.any(_li.take(h.victims,r,foe),axis=-1),'sleep_clause_blocks_opp':j.any(_li.take(h.victims,r,owner),axis=-1),
            'self_hazard_credit':_li.take(hazard,r,owner,1),'opp_hazard_credit':_li.take(hazard,r,foe,0)}
        stats={**({}if stats is None else stats),'blocked_on_our_attack':_li.take(h.blocked,r,owner),'pursuit_intercept_predict':_li.take(h.pursuit,r,foe)}
        reveals=_li.take(h.weather,r,foe).reshape(b,6)
        stats.update({'weather_reveal_'+str(i):reveals[:,i]for i in range(6)})
    return PublicView(_li.take(trackers.own_order,r,owner),take('species'),take('level'),take('gender'),
        take('hp_percent').astype(j.float32)/100.0,take('status'),active,present,take('fainted'),
        j.zeros((b,6,2),j.int32) if trackers.shown_types is None else take('shown_types'),take('boosts'),
        z if trackers.original_ability is None else take('original_ability'),
        z if trackers.shown_item is None else take('shown_item'),
        trackers.error|j.where((observer_sides<0)|(observer_sides>1),BAD_OBSERVER,0),None,
        trackers.turn,trackers.weather,trackers.spikes,belief_mask,optional('observed_moves'),
        optional('pp_used'),optional('last_moves'),optional('borrowed_ability'),own_last,mon,own_mon,fields,own_borrowed,optional('public_volatiles'),own_vol,optional('item_removed'),own_removed,stats,
        None if visible is None else _li.take(visible.visible_pin,r,foe),None if visible is None else _li.take(visible.visible_pin_active,r,foe))


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
