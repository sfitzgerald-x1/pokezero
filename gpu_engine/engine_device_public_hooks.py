"""Opt-in device public writes alongside the original numerical transition.

Each hook consumes operands from its actual operation, never selected actions
or a sampled opponent world. This is a knowledge carry, not an event log or a
capture/admission system. Hard source masks are not refilled on inconsistency.
"""
from typing import NamedTuple
import jax
import jax.numpy as j
from . import engine_device_trackers as tr
from .public_active import NAMES as VOLATILE_NAMES

MON_KEYS=('turns_active','turns_active_total','sleep_turns','rest_sleep','toxic_stage',
          'confusion_turns','encore_turns','stall_counter','sub_present','meanlook_trap',
          'last_damage_dealt','last_damage_taken','switched_before_attack','stayed_and_attacked',
          'moves_shown','switches','item_swapped')
DEFERRED={
 'wrap_trap_turns':'No supported numerical wrap-duration handler.',
 'choice_locked':'Opponent choice lock is not publicly revealed by engine state alone.',
 'truant_loaf':'Private clock is not polled; public activation hook not implemented.',
 'sleepClause':'Needs original public causal attribution, not private party status polling.',
 'hazard_credit':'Hazard exposure is public; causal setter credit not yet retained.',
 'tendency_outcomes':'Blocked-attack/pursuit/weather attribution remains deferred; basic switch/stay/opportunity counters are public.',
 'confusion_self_damage':'Ordinary direct-damage fractions are recorded; confusion self-hit damage metric remains deferred.',
 'item_swap_transform':'No corresponding numerical engine handler in this transition; item_swapped stays zero.',
 'item_credit':'Consumed native items are retained/revealed, but removal attribution is not implemented.',
}

class Context:
    # Trace-local builder only. No shared/global object and no validity cache;
    # wrappers return the immutable arrays, never this Python object.
    __slots__=('trackers','belief_tables')
    def __init__(self,trackers,belief_tables=None):
        self.trackers=trackers;self.belief_tables=belief_tables
    def _replace(self,**values):
        for key,value in values.items():setattr(self,key,value)
        return self

class PublicStepResult(NamedTuple):
    step:object
    trackers:object

class Initialized(NamedTuple):
    state:object
    trackers:object


def _r(state):return j.arange(state.hp.shape[0])
def _get(state,name,side):return getattr(state,name)[_r(state),side,state.active[_r(state),side]]
def _mask(mask,b):return j.broadcast_to(j.asarray(mask,j.bool_),(b,))

def _write(t,name,side,physical,value,use):
    r=j.arange(len(t.error));p=j.clip(physical,0,5);slot=t.physical_to_public[r,side,p];at=j.clip(slot,0,5)
    use=use&(physical>=0)&(physical<6)&(slot>=0)&(slot<6)
    old=getattr(t,name);v=old[r,side,at]
    gate=use.reshape(use.shape+(1,)*(v.ndim-1))
    return t._replace(**{name:old.at[r,side,at].set(j.where(gate,value,v))})

def feature(c,side,physical,name,value,use,*,add=False):
    t=c.trackers;r=j.arange(len(t.error));slot=t.physical_to_public[r,side,j.clip(physical,0,5)];at=j.clip(slot,0,5)
    use=use&(slot>=0)&(slot<6)&(physical>=0)&(physical<6)
    values=t.mon_features;old=values[name];v=old[r,side,at]
    new=j.where(use,v+value if add else value,v)
    if name in ('switched_before_attack','stayed_and_attacked','turns_active_total'):new=j.minimum(64,new)
    return c._replace(trackers=t._replace(mon_features={**values,name:old.at[r,side,at].set(new)}))

def _narrow(c,side,physical,fact,use,kind):
    if c.belief_tables is None:return c
    t=c.trackers;r=j.arange(len(t.error));p=j.clip(physical,0,5);slot=t.physical_to_public[r,side,p];at=j.clip(slot,0,5)
    species=t.species[r,side,at];table=getattr(c.belief_tables,kind+'_mask')
    good=(species>=0)&(species<table.shape[0])&(fact>=0)&(fact<table.shape[1])
    bits=table[j.clip(species,0,table.shape[0]-1),j.clip(fact,0,table.shape[1]-1)]
    bits=j.where(good[:,None],bits,j.uint32(0))
    old=t.belief_mask[r,side,at]
    # Zero means no public fact, not evidence for a missing source variant.
    bits=j.where((fact==0)[:,None],j.uint32(0xffffffff),bits)
    use=use&(slot>=0)&(slot<6)
    return c._replace(trackers=t._replace(belief_mask=t.belief_mask.at[r,side,at].set(j.where(use[:,None],old&bits,old))))


def initialize(state,belief_tables=None):
    t=tr.initialize(state);b=len(t.error);z=j.zeros((b,2,6),j.int32)
    t=t._replace(observed_moves=j.zeros((b,2,6,4),j.int32),pp_used=j.zeros((b,2,6,4),j.int32),
        last_moves=z,tendencies=j.zeros((b,2,6,2),j.int32),belief_mask=j.zeros((b,2,6,2),j.uint32),
        shown_ability=z,original_ability=z,shown_item=z,item_removed=j.zeros_like(z,bool),shown_types=j.zeros((b,2,6,2),j.int32),borrowed_ability=z,
        mon_features={key:(z.astype(j.float32) if key.startswith('last_damage') else z) for key in MON_KEYS},
        damage_exact_dealt=z.astype(j.float32),damage_exact_taken=z.astype(j.float32),
        tendency=tr.Tendency(j.zeros((b,2),bool),j.full((b,2),-1,j.int32),j.zeros((b,2),j.int32),
            j.zeros((b,2),j.int32),j.zeros((b,2,6,6,2),j.int32)),
        public_volatiles=j.zeros((b,2,6,len(VOLATILE_NAMES)),bool),field_features={'weather_upkeeps':j.zeros(b,j.int32),
            'weather_permanent':j.zeros(b,bool),'wish_turns':j.zeros((b,2),j.int32),'wish_pending':j.zeros((b,2),bool)})
    c=Context(t,belief_tables)
    for side in range(2):c=install(c,state,side,j.ones(b,bool),initial=True)
    # Original startup emits switches, then TURN1 before the first request.
    return public_turn(c,state,~state.done,state.turn+1)


def public_turn(c,state,use,turn):
    """Actual next ordinary-request boundary, not the private engine counter."""
    t=c.trackers
    for side in range(2):
        p=state.active[_r(state),side]
        for name in ('turns_active','turns_active_total'):
            c=feature(c,side,p,name,j.ones_like(p),use,add=True)
    return c._replace(trackers=c.trackers._replace(turn=j.where(use,turn,t.turn)))


def opportunity(c,side,use):
    t=c.trackers;td=t.tendency;r=j.arange(len(t.error))
    fresh=use&(td.opportunity_turn[r,side]!=t.turn)
    turns=td.opportunity_turn.at[r,side].set(j.where(use,t.turn,td.opportunity_turn[r,side]))
    counts=td.opportunities.at[r,side].set(j.minimum(64,td.opportunities[r,side]+fresh))
    return c._replace(trackers=t._replace(tendency=td._replace(opportunity_turn=turns,opportunities=counts)))


def switch_decision(c,state,side,use,*,baton):
    """Same actual switch before departure; forced/Baton are not decisions."""
    t=c.trackers;td=t.tendency;r=_r(state);p=state.active[r,side]
    voluntary=use&~baton&(_get(state,'hp',side)>0)
    bail=voluntary&~td.moved[r,side]
    c=feature(c,side,p,'switched_before_attack',j.ones_like(p),bail,add=True)
    c=matchup(c,side,bail,0)
    t=c.trackers;td=t.tendency
    counts=td.switches.at[r,side].set(j.minimum(64,td.switches[r,side]+voluntary))
    c=c._replace(trackers=t._replace(tendency=td._replace(switches=counts)))
    return opportunity(c,side,voluntary)


def matchup(c,side,use,kind):
    t=c.trackers;td=t.tendency;r=j.arange(len(t.error));a=t.active_slot[r,side];f=t.active_slot[r,1-side]
    good=use&(a>=0)&(f>=0);a=j.clip(a,0,5);f=j.clip(f,0,5)
    old=td.matchup[r,side,a,f,kind]
    values=td.matchup.at[r,side,a,f,kind].set(j.minimum(8,old+good))
    return c._replace(trackers=t._replace(tendency=td._replace(matchup=values)))


def install(c,state,side,use,*,initial=False):
    t=c.trackers;b=len(t.error);use=_mask(use,b);r=j.arange(b);p=state.active[r,side]
    old=t.active_physical[r,side];fresh=t.physical_to_public[r,side,j.clip(p,0,5)]<0
    if not initial:
        # One actual installation, including intermediate drag/Baton installs.
        order=t.own_order[r,side];position=j.argmax(order==p[:,None],axis=1)
        swapped=order.at[r,position].set(order[:,0]).at[:,0].set(p)
        t=t._replace(own_order=t.own_order.at[r,side].set(j.where(use[:,None],swapped,order)))
    t=tr._install(t,state,side,use);slot=t.active_slot[r,side];at=j.clip(slot,0,5)
    c=c._replace(trackers=t)
    if c.belief_tables is not None:
        species=_get(state,'species',side);tab=c.belief_tables.species_initial_mask
        known=(species>0)&(species<tab.shape[0]);bits=tab[j.clip(species,0,tab.shape[0]-1)]
        bits=j.where(known[:,None],bits,j.uint32(0))
        prev=c.trackers.belief_mask[r,side,at]
        # Initial call already installed the active encounter in tr.initialize.
        first=use&(fresh|j.asarray(initial))
        c=c._replace(trackers=_write(c.trackers,'belief_mask',side,p,j.where(first[:,None],bits,prev),use))
        if hasattr(c.belief_tables,'species_types'):
            types=c.belief_tables.species_types[j.clip(species,0,tab.shape[0]-1)]
            c=c._replace(trackers=_write(c.trackers,'shown_types',side,p,j.where(known[:,None],types,0),use))
    # switch resets public volatiles, not cumulative shown move/PP knowledge.
    for name in ('turns_active','confusion_turns','encore_turns','stall_counter','sub_present','meanlook_trap'):
        c=feature(c,side,p,name,j.zeros(b,j.int32),use)
    c=c._replace(trackers=_write(c.trackers,'last_moves',side,p,j.full((b,),-1,j.int32),use))
    td=c.trackers.tendency
    c=c._replace(trackers=c.trackers._replace(tendency=td._replace(
        moved=td.moved.at[r,side].set(j.where(use,False,td.moved[r,side])))))
    c=c._replace(trackers=_write(c.trackers,'public_volatiles',side,p,j.zeros((b,len(VOLATILE_NAMES)),bool),use))
    if not initial:c=feature(c,side,p,'switches',j.ones(b,j.int32),use,add=True)
    return c


def observe(c,state,use):
    """Actual current public HP/status/stages/field only; never poll benches."""
    use=_mask(use,len(c.trackers.error));r=_r(state)
    for side in range(2):
        t=c.trackers;p=state.active[:,side];slot=j.clip(t.physical_to_public[r,side,p],0,5)
        status=_get(state,'status',side);changed=use&(t.status[r,side,slot]!=status)
        for key in ('sleep_turns','rest_sleep','toxic_stage'):
            c=feature(c,side,p,key,j.zeros_like(p),changed)
        c=c._replace(trackers=tr._observe(c.trackers,state,side,p,use,identity=False))
    t=c.trackers
    return c._replace(trackers=t._replace(weather=j.where(use,state.weather,t.weather),
        spikes=j.where(use[:,None],state.spikes,t.spikes)))


def reveal_ability(c,state,side,use,*,ability=None,original=True):
    p=state.active[_r(state),side];ability=_get(state,'ability',side) if ability is None else ability
    use=_mask(use,len(c.trackers.error));r=_r(state);slot=j.clip(c.trackers.physical_to_public[r,side,p],0,5)
    borrowed=c.trackers.borrowed_ability[r,side,slot]
    c=c._replace(trackers=_write(c.trackers,'shown_ability',side,p,ability,use))
    if original:
        c=c._replace(trackers=_write(c.trackers,'original_ability',side,p,ability,use&(borrowed==0)))
        c=_narrow(c,side,p,ability,use&(borrowed==0),'ability')
    return c


def reveal_item(c,state,side,item,use,*,original=True):
    p=state.active[_r(state),side];use=_mask(use,len(c.trackers.error))
    c=c._replace(trackers=_write(c.trackers,'shown_item',side,p,item,use))
    return _narrow(c,side,p,item,use,'item') if original else c


def entry(c,before,after,side,table,use):
    c=observe(c,after,use)
    if table is None:return c
    ability=_get(before,'ability',side);starts=use&(_get(after,'hp',side)>0)
    reveal=starts&((table.weather_ability[ability]!=0)|table.intimidate[ability])
    c=reveal_ability(c,after,side,reveal,ability=ability)
    blocked=starts&table.intimidate[ability]&(_get(before,'hp',1-side)>0)&(_get(before,'substitute_hp',1-side)==0)&table.attack_drop_immune[_get(before,'ability',1-side)]
    c=reveal_ability(c,before,1-side,blocked)
    changed=starts&(table.weather_ability[ability]!=0)&((before.weather!=after.weather)|(before.weather_counter!=after.weather_counter))
    c=weather(c,before,after,changed,permanent=True)
    if table.trace is not None:
        copied=starts&table.trace[ability]&(_get(before,'hp',1-side)>0)
        c=reveal_ability(c,before,side,copied,ability=ability)
        c=reveal_ability(c,before,1-side,copied)
        p=after.active[_r(after),side]
        c=c._replace(trackers=_write(c.trackers,'borrowed_ability',side,p,_get(after,'ability',side),copied))
        c=reveal_ability(c,after,side,copied,original=False)
    current=_get(after,'ability',side)
    cure=starts&(_get(before,'status',side)!=0)&(_get(after,'status',side)==0)
    if table.own_tempo is not None:
        cure |= starts&table.own_tempo[current]&(_get(before,'confusion_counter',side)>0)
    c=reveal_ability(c,after,side,cure)
    # Pressure's Gen3 start line is owner-only; never shared with the foe here.
    return c


def depart(c,before,after,side,table,use):
    p=before.active[_r(before),side]
    if table is not None:
        cure=use&(_get(before,'hp',side)>0)&(_get(before,'status',side)!=0)&table.natural_cure[_get(before,'ability',side)]
        c=reveal_ability(c,before,side,cure)
        c=observe(c,after,cure)
    # A borrowed ability ends silently; known original identity is not guessed.
    c=c._replace(trackers=_write(c.trackers,'shown_ability',side,p,c.trackers.original_ability[_r(before),side,j.clip(c.trackers.active_slot[_r(before),side],0,5)],use&(c.trackers.borrowed_ability[_r(before),side,j.clip(c.trackers.active_slot[_r(before),side],0,5)]!=0)))
    c=c._replace(trackers=_write(c.trackers,'borrowed_ability',side,p,j.zeros_like(use,j.int32),use))
    return c


def move(c,state,side,move,use,*,pp_use=False,original=True,count_turn=True,preparation=False):
    """Actual declarations charge public PP, never hidden physical PP cost.

    Original source target semantics and only publicly running Pressure can
    double a caller charge. Called Sleep Talk children remain uncharged.
    """
    actual_move=move
    if c.belief_tables is not None and hasattr(c.belief_tables,'public_move'):
        valid=(move>=0)&(move<len(c.belief_tables.public_move))
        at=j.clip(move,0,len(c.belief_tables.public_move)-1)
        evidence=valid&c.belief_tables.source_move_evidence[at]
        move=j.where(valid,c.belief_tables.public_move[at],0)
    else:evidence=j.ones_like(use)
    t=c.trackers;r=_r(state);p=state.active[r,side];slot=j.clip(t.physical_to_public[r,side,p],0,5)
    old=t.observed_moves[r,side,slot];has=j.any(old==move[:,None],axis=1);empty=j.argmax(old==0,axis=1)
    at=j.where(has,j.argmax(old==move[:,None],axis=1),empty)
    declared=use&(move>0)
    room=has|j.any(old==0,axis=1);use=declared&room
    previous=t.observed_moves[r,side,slot,at]
    observed=t.observed_moves.at[r,side,slot,at].set(j.where(use,move,previous))
    count=t.pp_used[r,side,slot,at]
    extra=j.zeros_like(use)
    if (c.belief_tables is not None and hasattr(c.belief_tables,'public_foe_targeted')
            and hasattr(c.belief_tables,'public_pressure')):
        targets=c.belief_tables.public_foe_targeted;pressure=c.belief_tables.public_pressure
        foe=1-side;foe_slot=j.clip(t.active_slot[r,foe],0,5)
        copied=t.borrowed_ability[r,foe,foe_slot]
        running=j.where(copied!=0,copied,t.original_ability[r,foe,foe_slot])
        valid_ability=(running>0)&(running<len(pressure))
        known_pressure=valid_ability&pressure[j.clip(running,0,len(pressure)-1)]
        target_known=(actual_move>0)&(actual_move<len(targets))
        extra=known_pressure&target_known&targets[j.clip(actual_move,0,len(targets)-1)]
    charge=(use&j.asarray(pp_use)).astype(j.int32)*(1+extra.astype(j.int32))
    pp=t.pp_used.at[r,side,slot,at].set(count+charge)
    c=c._replace(trackers=t._replace(observed_moves=observed,pp_used=pp))
    # The child [from] line and preparation-only -singleturn annotation can
    # reveal a move, but neither is an ordinary MOVE header/decision datum.
    header=count_turn
    if header:c=c._replace(trackers=_write(c.trackers,'last_moves',side,p,move,declared))
    if not preparation:
        c=feature(c,side,p,'moves_shown',j.ones_like(move),declared,add=True)
    if header:
        t=c.trackers;td=t.tendency;first=declared&~td.moved[r,side]
        c=feature(c,side,p,'stayed_and_attacked',j.ones_like(move),first,add=True)
        c=matchup(c,side,first,1)
        t=c.trackers;td=t.tendency
        c=c._replace(trackers=t._replace(tendency=td._replace(moved=td.moved.at[r,side].set(td.moved[r,side]|declared))))
        c=opportunity(c,side,declared&(_get(state,'charge_move',side)==0))
    return _narrow(c,side,p,move,declared&evidence,'move') if original else c


def item_changes(c,before,after,use):
    for side in range(2):
        old=_get(before,'item',side);consumed=use&(old>0)&(_get(after,'item',side)==0)
        c=reveal_item(c,before,side,old,consumed)
        c=c._replace(trackers=_write(c.trackers,'item_removed',side,after.active[_r(after),side],j.ones_like(consumed),consumed))
    return observe(c,after,use)


def party_cure(c,before,after,side,use,table=None):
    # Heal Bell's actual living non-Soundproof write; known slots only. Never
    # reveal unseen bench identity or private status from a scan.
    if table is not None:c=reveal_ability(c,before,side,use&table.soundproof[_get(before,'ability',side)])
    r=_r(before)
    for p in range(6):
        changed=use&(before.status[r,side,p]!=after.status[r,side,p])
        old=c.trackers.status;slot=c.trackers.physical_to_public[r,side,p];at=j.clip(slot,0,5)
        yes=changed&(slot>=0)
        c=c._replace(trackers=c.trackers._replace(status=old.at[r,side,at].set(j.where(yes,after.status[r,side,p],old[r,side,at]))))
        cured=yes&(after.status[r,side,p]==0)
        for name in ('sleep_turns','rest_sleep','toxic_stage'):
            c=feature(c,side,p,name,j.zeros(len(r),j.int32),cured)
    return c


def damage(c,state,side,dealt,taken,use):
    # Keep observer-relative fractions; only readout of the OWN victim uses
    # exact HP. Foe-victim public fractions come from shown percentage points.
    r=_r(state);foe=1-side;p=state.active[r,foe];slot=j.clip(c.trackers.physical_to_public[r,foe,p],0,5)
    previous=c.trackers.hp_percent[r,foe,slot]
    current=tr._percent(_get(state,'hp',foe),_get(state,'max_hp',foe))
    shown=j.maximum(0,previous-current).astype(j.float32)/100.0
    exact=taken.astype(j.float32)/j.maximum(_get(state,'max_hp',foe),1).astype(j.float32)
    c=c._replace(trackers=_write(c.trackers,'damage_exact_dealt',side,state.active[r,side],exact,use))
    c=c._replace(trackers=_write(c.trackers,'damage_exact_taken',foe,p,exact,use))
    c=feature(c,side,state.active[r,side],'last_damage_dealt',shown,use)
    return feature(c,foe,p,'last_damage_taken',shown,use)


def turn_end(c,before,after,ordinary):
    c=observe(c,after,~before.done)
    for side in range(2):
        p=after.active[_r(after),side];alive=ordinary&(_get(after,'hp',side)>0)
        for name,active in [('toxic_stage',_get(after,'status',side)==4),
                            ('confusion_turns',_get(after,'confusion_counter',side)>0),('encore_turns',_get(after,'encore_counter',side)>0)]:
            c=feature(c,side,p,name,j.ones_like(p),alive&active,add=True)
            c=feature(c,side,p,name,j.zeros_like(p),ordinary&~active)
    t=c.trackers;fields=t.field_features
    up=fields['weather_upkeeps']+(ordinary&(after.weather!=0)).astype(j.int32)
    c=c._replace(trackers=t._replace(field_features={**fields,'weather_upkeeps':up}))
    # A paused action/residual or forced replacement has no next TURN line.
    # A completed replacement may open that line even without residual work.
    ready=(~before.done&~after.done&~after.pending_residual&~j.any(after.forced_switch,axis=1)
        &(c.trackers.turn<after.turn+1))
    return public_turn(c,after,ready,after.turn+1)


def color_change(c,before,after,side,use):
    changed=use&j.any(_get(before,'types',side)!=_get(after,'types',side),axis=1)
    c=reveal_ability(c,after,side,changed)
    p=after.active[_r(after),side]
    return c._replace(trackers=_write(c.trackers,'shown_types',side,p,_get(after,'types',side),changed))


def drag(c,before,after,side,table,use):
    # One actual drag operation owns exactly one depart/install/entry. Its
    # pre-entry incoming private operands already exist in the actual before
    # state; only public handler results are exposed.
    c=depart(c,before,after,side,table,use)
    c=install(c,after,side,use)
    incoming=before._replace(active=before.active.at[_r(before),side].set(after.active[_r(after),side]))
    return entry(c,incoming,after,side,table,use) if table is not None else observe(c,after,use)


def public_move_state(c,before,after,side,move,use):
    """Observable volatile starts/ends, not private remaining durations."""
    for target in (side,1-side):
        p=after.active[_r(after),target]
        for key,old,new in (
            ('sub_present',_get(before,'substitute_hp',target)>0,_get(after,'substitute_hp',target)>0),
            ('confusion_turns',_get(before,'confusion_counter',target)>0,_get(after,'confusion_counter',target)>0),
            ('encore_turns',_get(before,'encore_counter',target)>0,_get(after,'encore_counter',target)>0),
            ('meanlook_trap',_get(before,'trap_source',target)>=0,_get(after,'trap_source',target)>=0)):
            changed=use&(old!=new)
            # Start at zero elapsed; active presence supplied separately for
            # duration features whose zero could also mean absent.
            value=new.astype(j.int32) if key in ('sub_present','meanlook_trap') else j.zeros_like(p)
            c=feature(c,target,p,key,value,changed)
    for target in (side,1-side):
        for name,field in (('confusion','confusion_counter'),('encore','encore_counter'),('substitute','substitute_hp'),('solarbeam','charge_move')):
            old=_get(before,field,target)>0;new=_get(after,field,target)>0
            c=volatile(c,after,target,name,new,use&(old!=new))
        old=(_get(before,'volatile',target)&j.uint32(4))!=0;new=(_get(after,'volatile',target)&j.uint32(4))!=0
        c=volatile(c,after,target,'leechseed',new,use&(old!=new))
    return observe(c,after,use)


def weather(c,before,after,use,*,permanent=False):
    t=c.trackers;fields=t.field_features
    changed=use&((before.weather!=after.weather)|(before.weather_counter!=after.weather_counter))
    c=c._replace(trackers=t._replace(field_features={**fields,
        'weather_upkeeps':j.where(changed,0,fields['weather_upkeeps']),
        'weather_permanent':j.where(changed,j.asarray(permanent)&(after.weather!=0),fields['weather_permanent'])}))
    return observe(c,after,use)


def wish(c,side,use,*,start=False):
    t=c.trackers;f=t.field_features;r=j.arange(len(t.error));old=f['wish_turns'][r,side];pending=f['wish_pending'][r,side]
    value=j.zeros_like(old) if start else old+1
    enabled=use if start else use&pending
    turns=f['wish_turns'].at[r,side].set(j.where(enabled,value,old))
    waiting=f['wish_pending'].at[r,side].set(j.where(enabled,j.asarray(start)|(value<2),pending))
    return c._replace(trackers=t._replace(field_features={**f,'wish_turns':turns,'wish_pending':waiting}))


def before_move(c,state,side,result,use):
    p=state.active[_r(state),side]
    c=feature(c,side,p,'sleep_turns',j.ones_like(p),use&(_get(state,'status',side)==5)&(result.status==5),add=True)
    return opportunity(c,side,use&~result.can_act&~result.confusion_self_hit)


def volatile(c,state,side,name,value,use):
    r=_r(state);p=state.active[r,side];slot=j.clip(c.trackers.physical_to_public[r,side,p],0,5)
    old=c.trackers.public_volatiles[r,side,slot];new=old.at[:,VOLATILE_NAMES.index(name)].set(value)
    return c._replace(trackers=_write(c.trackers,'public_volatiles',side,p,new,use))


def baton_copy(c,state,side,outgoing,use):
    r=_r(state);p=state.active[r,side];src=j.clip(c.trackers.physical_to_public[r,side,outgoing],0,5)
    names={'confusion','leechseed','substitute','solarbeam'}
    allowed=j.asarray([name in names for name in VOLATILE_NAMES])
    bag=c.trackers.public_volatiles[r,side,src]&allowed[None,:]
    c=c._replace(trackers=_write(c.trackers,'public_volatiles',side,p,bag,use))
    for name in ('confusion_turns','sub_present','meanlook_trap'):
        value=c.trackers.mon_features[name][r,side,src]
        c=feature(c,side,p,name,value,use)
    c=c._replace(trackers=_write(c.trackers,'last_moves',side,p,j.full_like(p,-2),use))
    return c


def withdraw(c,state,side,physical,use):
    b=len(c.trackers.error)
    c=c._replace(trackers=_write(c.trackers,'boosts',side,physical,j.zeros((b,7),j.int32),use))
    c=c._replace(trackers=_write(c.trackers,'public_volatiles',side,physical,j.zeros((b,len(VOLATILE_NAMES)),bool),use))
    for name in ('turns_active','confusion_turns','encore_turns','stall_counter','sub_present','meanlook_trap'):
        c=feature(c,side,physical,name,j.zeros(b,j.int32),use)
    return c
