"""Opt-in device public writes alongside the original numerical transition.

Each hook consumes operands from its actual operation, never selected actions
or a sampled opponent world. This is a knowledge carry, not an event log or a
capture/admission system. Hard source masks are not refilled on inconsistency.
"""
from typing import NamedTuple
import jax
import jax.numpy as j
from . import engine_device_trackers as tr
from . import engine_device_negative_items as negative_items
from .absorption import FLASH_FIRE
from .public_active import NAMES as VOLATILE_NAMES
from . import engine_device_damage_history as damage_history
from . import engine_device_public_port as port
from . import engine_device_visible_belief as visible_belief
from . import lane_index as _li

# Same-write HP notification classes for the separate public negative-item
# updater. These are static provenance labels, not an event/witness roster.
HP_ACTION,HP_PRE_ITEM,HP_POST_ITEM,HP_UNKNOWN=range(4)

MON_KEYS=('turns_active','turns_active_total','sleep_turns','rest_sleep','toxic_stage',
          'sleep_skipped_turns','sleep_cant_pending','sleep_usable_pending',
          'confusion_turns','encore_turns','stall_counter','sub_present','meanlook_trap',
          'last_damage_dealt','last_damage_taken','switched_before_attack','stayed_and_attacked',
          'moves_shown','switches','item_swapped',
          # Fidelity M1 (design 2.5). B2's per-stint active flags (public_active: ITEM_SET from a Trick -item line,
          # ITEM_END from any -enditem, CHOICE_MOVE from the holder's own move line; ARRIVE and FAINT reset):
          # item_swapped (column 41) beside them, choice_public and choice_locked (column 40). B2's per-mon item
          # ledger (public_items: mutated, original), which survives switches: item_mutated, item_original.
          'choice_public','choice_locked','item_mutated','item_original',
          # Fidelity M2 (design 2.5): B2's WRAP_TRAP_TURNS (column 33), public turns elapsed while partiallytrapped is
          # in the bag: +1 per TURN line, reset by -end (WRAP_END), a faint or an arrival without Baton Pass.
          'wrap_trap_turns')
# Per-stint flags: zeroed at every arrival (B2 ARRIVE), a faint and an -enditem (ITEM_END).
STINT_ITEM_KEYS=('item_swapped','choice_public','choice_locked')
DEFERRED={
 'confusion_self_damage':'Ordinary direct-damage fractions are recorded; confusion self-hit damage metric remains deferred.',
}

class Context:
    # Trace-local builder only. No shared/global object and no validity cache;
    # wrappers return the immutable arrays, never this Python object.
    __slots__=('trackers','belief_tables','port_tables')
    def __init__(self,trackers,belief_tables=None,port_tables=None):
        self.trackers=trackers;self.belief_tables=belief_tables;self.port_tables=port_tables
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
def _get(state,name,side):return _li.take(getattr(state,name),_r(state),side,_li.take(state.active,_r(state),side))
def _mask(mask,b):return j.broadcast_to(j.asarray(mask,j.bool_),(b,))

def _write(t,name,side,physical,value,use):
    r=j.arange(len(t.error));p=j.clip(physical,0,5);slot=_li.take(t.physical_to_public,r,side,p);at=j.clip(slot,0,5)
    use=use&(physical>=0)&(physical<6)&(slot>=0)&(slot<6)
    old=getattr(t,name);gate=use.reshape(use.shape+(1,)*(old.ndim-3))
    return t._replace(**{name:_li.update(old,r,(side,at),lambda v,gate,value:j.where(gate,value,v),gate,value)})

def feature(c,side,physical,name,value,use,*,add=False):
    t=c.trackers;r=j.arange(len(t.error));slot=_li.take(t.physical_to_public,r,side,j.clip(physical,0,5));at=j.clip(slot,0,5)
    use=use&(slot>=0)&(slot<6)&(physical>=0)&(physical<6)
    values=t.mon_features;old=values[name];capped=name in ('switched_before_attack','stayed_and_attacked','turns_active_total')
    def write(v,use,value):
        new=j.where(use,v+value if add else value,v)
        return j.minimum(64,new) if capped else new
    return c._replace(trackers=t._replace(mon_features={**values,name:_li.update(old,r,(side,at),write,use,value)}))

def _refresh(c,side,physical,use):
    """One actual subject's source upsert/replace; no endpoint-wide refresh."""
    physical=j.broadcast_to(j.asarray(physical,j.int32),(len(c.trackers.error),))
    return c._replace(trackers=visible_belief.refresh(c.trackers,c.belief_tables,side,physical,use))


def _narrow(c,side,physical,fact,use,kind):
    if c.belief_tables is None:return c
    t=c.trackers;r=j.arange(len(t.error));p=j.clip(physical,0,5);slot=_li.take(t.physical_to_public,r,side,p);at=j.clip(slot,0,5)
    species=_li.take(t.species,r,side,at);table=getattr(c.belief_tables,kind+'_mask')
    good=(species>=0)&(species<table.shape[0])&(fact>=0)&(fact<table.shape[1])
    bits=table[j.clip(species,0,table.shape[0]-1),j.clip(fact,0,table.shape[1]-1)]
    bits=j.where(good[:,None],bits,j.uint32(0))
    # Zero means no public fact, not evidence for a missing source variant.
    bits=j.where((fact==0)[:,None],j.uint32(0xffffffff),bits)
    use=use&(slot>=0)&(slot<6)
    c=c._replace(trackers=t._replace(belief_mask=_li.update(t.belief_mask,r,(side,at),lambda v,use,bits:j.where(use,v&bits,v),use[:,None],bits)))
    return _refresh(c,side,physical,use&(fact!=0))


def _reset_item_narrowing(c,side,physical,use):
    """B2 stops narrowing by a mutated item (design 13 item 4).

    public_items.matching_item gives key 0 once an item is mutated while
    item_belief_narrowing is False (src/pokezero/observation.py:382), so B2's
    full belief is rebuilt from the facts that still narrow it: the species
    pool, the original ability, the revealed moves and the standing ability/
    item exclusions. _narrow's AND cannot be undone, so the subject's hard mask
    is recomputed from exactly those retained facts (the same tables and gates
    as _narrow and visible_belief.exclude), then its visible basis refreshed as
    B2's mutation callback does.
    """
    if c.belief_tables is None or c.trackers.belief_mask is None:return c
    t=c.trackers;tables=c.belief_tables;b=len(t.error);r=j.arange(b)
    physical=j.broadcast_to(j.asarray(physical,j.int32),(b,))
    slot=_li.take(t.physical_to_public,r,side,j.clip(physical,0,5));at=j.clip(slot,0,5)
    use=_mask(use,b)&(slot>=0)&(slot<6)&(physical>=0)&(physical<6)
    pools=tables.species_initial_mask;species=_li.take(t.species,r,side,at)
    known=(species>0)&(species<pools.shape[0]);sp=j.clip(species,0,pools.shape[0]-1)
    hard=j.where(known[:,None],pools[sp],j.uint32(0))
    def narrowed(kind,fact):
        table=getattr(tables,kind+'_mask')
        good=(species>=0)&(species<table.shape[0])&(fact>=0)&(fact<table.shape[1])
        bits=j.where(good[:,None],table[j.clip(species,0,table.shape[0]-1),j.clip(fact,0,table.shape[1]-1)],j.uint32(0))
        return j.where((fact==0)[:,None],j.uint32(0xffffffff),bits)
    if t.original_ability is not None:hard&=narrowed('ability',_li.take(t.original_ability,r,side,at))
    if t.observed_moves is not None:
        moves=_li.take(t.observed_moves,r,side,at)
        for k in range(moves.shape[-1]):hard&=narrowed('move',moves[:,k])
    carry=t.visible_belief
    if carry is not None:
        for kind,ledger in (('ability',carry.excluded_abilities),('item',carry.excluded_items)):
            table=getattr(tables,kind+'_mask');names=_li.take(ledger,r,side,at)
            ids=j.arange(table.shape[1])
            excluded=((names[:,ids//32]>>(ids%32).astype(j.uint32)[None,:])&j.uint32(1))!=0
            rows=table[j.clip(species,0,table.shape[0]-1)]
            hard&=~j.bitwise_or.reduce(j.where(excluded[...,None],rows,j.uint32(0)),axis=1)
    c=c._replace(trackers=t._replace(belief_mask=_li.update(t.belief_mask,r,(side,at),
        lambda v,use,hard:j.where(use,hard,v),use[:,None],hard)))
    return _refresh(c,side,physical,use)


def initialize(state,belief_tables=None,*,port_tables=None):
    t=tr.initialize(state);b=len(t.error);z=j.zeros((b,2,6),j.int32)
    t=t._replace(observed_moves=j.zeros((b,2,6,4),j.int32),pp_used=j.zeros((b,2,6,4),j.int32),
        negative_items=negative_items.empty(b),
        last_moves=z,tendencies=j.zeros((b,2,6,2),j.int32),belief_mask=j.zeros((b,2,6,2),j.uint32),
        shown_ability=z,original_ability=z,shown_item=z,item_removed=j.zeros_like(z,bool),shown_types=j.zeros((b,2,6,2),j.int32),borrowed_ability=z,
        mon_features={key:(z.astype(j.float32) if key.startswith('last_damage') else z) for key in MON_KEYS},
        damage_exact_dealt=z.astype(j.float32),damage_exact_taken=z.astype(j.float32),
        tendency=tr.Tendency(j.zeros((b,2),bool),j.full((b,2),-1,j.int32),j.zeros((b,2),j.int32),
            j.zeros((b,2),j.int32),j.zeros((b,2,6,6,2),j.int32)),
        public_volatiles=j.zeros((b,2,6,len(VOLATILE_NAMES)),bool),field_features={'weather_upkeeps':j.zeros(b,j.int32),
            'weather_permanent':j.zeros(b,bool),'wish_turns':j.zeros((b,2),j.int32),'wish_pending':j.zeros((b,2),bool),
            'wish_set_turn':j.full((b,2),-1,j.int32),'items_removed_credit':j.zeros((b,2),j.int32)})
    t=t._replace(damage_history=damage_history.initialize(state),public_port=port.initialize(b))
    c=Context(t,belief_tables,port_tables)
    for side in range(2):c=install(c,state,side,j.ones(b,bool),initial=True)
    # Original startup emits switches, then TURN1 before the first request.
    return public_turn(c,state,~state.done,state.turn+1)


def public_turn(c,state,use,turn):
    """Actual next ordinary-request boundary, not the private engine counter."""
    t=c.trackers
    c=c._replace(trackers=port.turn(c.trackers,use,turn))
    for side in range(2):
        p=_li.take(state.active,_r(state),side)
        for name in ('turns_active','turns_active_total'):
            c=feature(c,side,p,name,j.ones_like(p),use,add=True)
        # Original public Toxic stage advances on TURN, not on residual work.
        t=c.trackers;slot=j.clip(_li.take(t.active_slot,_r(state),side),0,5)
        stage=_li.take(t.mon_features['toxic_stage'],_r(state),side,slot)
        c=feature(c,side,p,'toxic_stage',j.minimum(16,stage+1),use&(stage>0))
    if c.trackers.damage_history is not None:
        history,taken,dealt=damage_history.settle(c.trackers.damage_history,use)
        c=c._replace(trackers=c.trackers._replace(damage_history=history))
        for side in range(2):
            p=_li.take(state.active,_r(state),side)
            for name,value in (('last_damage_taken',taken[:,side,0]),('last_damage_dealt',dealt[:,side,0])):
                c=feature(c,side,p,name,value,use)
            for name,value in (('damage_exact_taken',taken[:,side,1]),('damage_exact_dealt',dealt[:,side,1])):
                c=c._replace(trackers=_write(c.trackers,name,side,p,value,use))
    c=c._replace(trackers=c.trackers._replace(turn=j.where(use,turn,t.turn)))
    # Original TURN replaces each living active belief, even if hard bits did
    # not change. A snapshot read itself never performs this operation.
    for side in range(2):
        slot=j.clip(_li.take(c.trackers.active_slot,_r(state),side),0,5)
        c=_refresh(c,side,_li.take(state.active,_r(state),side),use&~_li.take(c.trackers.fainted,_r(state),side,slot))
    return c


def opportunity(c,side,use):
    t=c.trackers;td=t.tendency;r=j.arange(len(t.error))
    fresh=use&(_li.take(td.opportunity_turn,r,side)!=t.turn)
    turns=_li.put(td.opportunity_turn,r,(side,),j.where(use,t.turn,_li.take(td.opportunity_turn,r,side)))
    counts=_li.put(td.opportunities,r,(side,),j.minimum(64,_li.take(td.opportunities,r,side)+fresh))
    return c._replace(trackers=t._replace(tendency=td._replace(opportunity_turn=turns,opportunities=counts)))


def switch_decision(c,state,side,use,*,baton):
    """Same actual switch before departure; forced/Baton are not decisions."""
    t=c.trackers;td=t.tendency;r=_r(state);p=_li.take(state.active,r,side)
    voluntary=use&~baton&(_get(state,'hp',side)>0)
    bail=voluntary&~_li.take(td.moved,r,side)
    c=feature(c,side,p,'switched_before_attack',j.ones_like(p),bail,add=True)
    c=matchup(c,side,bail,0)
    t=c.trackers;td=t.tendency
    counts=_li.put(td.switches,r,(side,),j.minimum(64,_li.take(td.switches,r,side)+voluntary))
    c=c._replace(trackers=t._replace(tendency=td._replace(switches=counts)))
    return opportunity(c,side,voluntary)


def matchup(c,side,use,kind):
    t=c.trackers;td=t.tendency;r=j.arange(len(t.error));a=_li.take(t.active_slot,r,side);f=_li.take(t.active_slot,r,1-side)
    good=use&(a>=0)&(f>=0);a=j.clip(a,0,5);f=j.clip(f,0,5)
    old=_li.take(td.matchup,r,side,a,f,kind)
    values=_li.put(td.matchup,r,(side,a,f,kind),j.minimum(8,old+good))
    return c._replace(trackers=t._replace(tendency=td._replace(matchup=values)))


def install(c,state,side,use,*,initial=False):
    t=c.trackers;b=len(t.error);use=_mask(use,b);r=j.arange(b);p=_li.take(state.active,r,side)
    old=_li.take(t.active_physical,r,side);fresh=_li.take(t.physical_to_public,r,side,j.clip(p,0,5))<0
    if not initial:
        # One actual installation, including intermediate drag/Baton installs.
        order=_li.take(t.own_order,r,side);position=j.argmax(order==p[:,None],axis=1)
        swapped=_li.put(order,r,(position,),order[:,0]).at[:,0].set(p)
        t=t._replace(own_order=_li.put(t.own_order,r,(side,),j.where(use[:,None],swapped,order)))
    t=tr._install(t,state,side,use);slot=_li.take(t.active_slot,r,side);at=j.clip(slot,0,5)
    # A newly revealed member already holds its public slot here; every actual
    # install (switch, replacement, drag, Baton target) moves it to the front.
    if not initial:t=tr.public_order_install(t,side,use)
    t=port.arrival(t,c.port_tables,state,side,use)
    t=negative_items.hp_line(t,state,side,use,source_class=negative_items.HP_ACTION)
    c=c._replace(trackers=t)
    if c.trackers.damage_history is not None:
        c=c._replace(trackers=c.trackers._replace(damage_history=damage_history.arrive(
            c.trackers.damage_history,state,side,use)))
        for name in ('last_damage_dealt','last_damage_taken'):
            c=feature(c,side,p,name,j.zeros(b,j.float32),use)
        for name in ('damage_exact_dealt','damage_exact_taken'):
            c=c._replace(trackers=_write(c.trackers,name,side,p,j.zeros(b,j.float32),use))
    if c.belief_tables is not None:
        species=_get(state,'species',side);tab=c.belief_tables.species_initial_mask
        known=(species>0)&(species<tab.shape[0]);bits=tab[j.clip(species,0,tab.shape[0]-1)]
        bits=j.where(known[:,None],bits,j.uint32(0))
        prev=_li.take(c.trackers.belief_mask,r,side,at)
        # Initial call already installed the active encounter in tr.initialize.
        first=use&(fresh|j.asarray(initial))
        c=c._replace(trackers=_write(c.trackers,'belief_mask',side,p,j.where(first[:,None],bits,prev),use))
        if hasattr(c.belief_tables,'species_types'):
            types=c.belief_tables.species_types[j.clip(species,0,tab.shape[0]-1)]
            c=c._replace(trackers=_write(c.trackers,'shown_types',side,p,j.where(known[:,None],types,0),use))
    # switch resets public volatiles, not cumulative shown move/PP knowledge.
    t=c.trackers;at=j.clip(_li.take(t.active_slot,r,side),0,5)
    count=_li.take(t.mon_features['sleep_turns'],r,side,at)
    skipped=_li.take(t.mon_features['sleep_skipped_turns'],r,side,at)
    c=feature(c,side,p,'sleep_turns',j.maximum(0,count-skipped),use&(_get(state,'status',side)==5))
    for name in ('turns_active','confusion_turns','encore_turns','stall_counter','sub_present','meanlook_trap',
                 'sleep_skipped_turns','sleep_cant_pending','sleep_usable_pending','toxic_stage'):
        c=feature(c,side,p,name,j.zeros(b,j.int32),use)
    # Fidelity M2: ARRIVE resets the partial-trap clock unless Baton Pass carries the bag entry (baton_copy follows).
    if 'wrap_trap_turns' in c.trackers.mon_features:c=feature(c,side,p,'wrap_trap_turns',j.zeros(b,j.int32),use)
    # Fidelity M1: B2's ARRIVE resets the seat's Trick/Choice flags (the item ledger stays per mon).
    for name in STINT_ITEM_KEYS:
        if name in c.trackers.mon_features:c=feature(c,side,p,name,j.zeros(b,j.int32),use)
    # Every arrival (switch, replacement, drag, Baton target) clears the public Mean
    # Look flag on BOTH seats, matching B2 (public_active.py:246, ARRIVE ignores
    # baton_pass) and the champion parser (src/pokezero/showdown.py:2752-2770).
    # Known B2 quirk, kept deliberately (lead decision, as for the stale recharge
    # bit): Showdown is right to release the trap on a switch or drag (the linked
    # trapped/trapper volatiles drop), but gen3 keeps it through Baton Pass in
    # both directions (gen4 conditions trapped/trapper noCopy:false, copied and
    # relinked by sim/pokemon.ts copyVolatileFrom:1247-1262), and the engine
    # follows Showdown there (baton.py:29-35). baton_copy therefore never carries it.
    c=feature(c,1-side,_li.take(state.active,r,1-side),'meanlook_trap',j.zeros(b,j.int32),use)
    c=c._replace(trackers=_write(c.trackers,'last_moves',side,p,j.full((b,),-1,j.int32),use))
    td=c.trackers.tendency
    c=c._replace(trackers=c.trackers._replace(tendency=td._replace(
        moved=_li.put(td.moved,r,(side,),j.where(use,False,_li.take(td.moved,r,side))))))
    c=c._replace(trackers=_write(c.trackers,'public_volatiles',side,p,j.zeros((b,len(VOLATILE_NAMES)),bool),use))
    if not initial:c=feature(c,side,p,'switches',j.ones(b,j.int32),use,add=True)
    return _refresh(c,side,p,use)


def observe(c,state,use,*,record_damage=True,rest_side=None,rest_mask=False):
    """Actual current public HP/status/stages/field only; never poll benches."""
    use=_mask(use,len(c.trackers.error));r=_r(state)
    if c.trackers.damage_history is not None:
        c=c._replace(trackers=c.trackers._replace(damage_history=damage_history.observe(
            c.trackers.damage_history,state,use,tr._percent,record_damage=record_damage)))
    for side in range(2):
        t=c.trackers;p=state.active[:,side];slot=j.clip(_li.take(t.physical_to_public,r,side,p),0,5)
        status=_get(state,'status',side);changed=use&(_li.take(t.status,r,side,slot)!=status)
        hp=_get(state,'hp',side);maximum=_get(state,'max_hp',side)
        valid=(maximum>0)&(hp>=0)&(hp<=maximum)&(status>=0)&(status<=6)
        condition_changed=changed|(use&(_li.take(t.hp_percent,r,side,slot)!=tr._percent(hp,maximum)))
        rest=j.zeros_like(use)if rest_side is None else (j.asarray(rest_side)==side)&j.asarray(rest_mask)
        c=c._replace(trackers=port.condition(t,side,p,_li.take(t.status,r,side,slot),status,use,
            rest=rest,faint=_get(state,'hp',side)==0))
        for key in ('sleep_turns','rest_sleep','sleep_skipped_turns','sleep_cant_pending','sleep_usable_pending'):
            c=feature(c,side,p,key,j.zeros_like(p),changed)
        c=feature(c,side,p,'toxic_stage',j.where(status==4,1,0),changed)
        c=feature(c,side,p,'toxic_stage',j.zeros_like(p),use&(hp==0))
        c=c._replace(trackers=tr._observe(c.trackers,state,side,p,use,identity=False))
        c=_refresh(c,side,p,condition_changed&valid)
    t=c.trackers
    return c._replace(trackers=t._replace(weather=j.where(use,state.weather,t.weather),
        spikes=j.where(use[:,None],state.spikes,t.spikes)))


def hp_line(c,state,side,use,*,source_class,record_damage=True):
    """Actual HP-line write cutpoint; use is original line emission eligibility.

    Notify the public-only negative-item updater BEFORE observe updates the
    public HP snapshot. No classification is guessed
    from a final state or from a shown-percent difference. Pain Split's -sethp
    updates the baseline with record_damage=False; self-faint has no HP line.
    """
    c=negative_hp_line(c,state,side,use,source_class=source_class)
    c=observe(c,state,use,record_damage=record_damage)
    if source_class==HP_POST_ITEM:
        # A newly arrived poisoned member starts at known public stage0.
        # Its first actual living poison-residual line exposes stage1; STATUS
        # already initialized other new poison to1. Never tick here again.
        r=_r(state);p=_li.take(state.active,r,side);slot=j.clip(_li.take(c.trackers.active_slot,r,side),0,5)
        stage=_li.take(c.trackers.mon_features['toxic_stage'],r,side,slot)
        hp=_get(state,'hp',side);maximum=_get(state,'max_hp',side)
        first=_mask(use,len(r))&(_get(state,'status',side)==4)&(stage==0)&(hp>0)&(maximum>=hp)
        c=feature(c,side,p,'toxic_stage',j.ones_like(p),first)
    # A literal HP/-sethp line upserts its subject even when the rounded
    # percentage is unchanged. Never refresh the other seat just because this
    # operation also polls its endpoint.
    hp=_get(state,'hp',side);maximum=_get(state,'max_hp',side)
    return _refresh(c,side,_li.take(state.active,_r(state),side),_mask(use,len(c.trackers.error))&(maximum>0)&(hp>=0)&(hp<=maximum))


def reveal_ability(c,state,side,use,*,ability=None,original=True):
    p=_li.take(state.active,_r(state),side);ability=_get(state,'ability',side) if ability is None else ability
    use=_mask(use,len(c.trackers.error));r=_r(state);slot=j.clip(_li.take(c.trackers.physical_to_public,r,side,p),0,5)
    borrowed=_li.take(c.trackers.borrowed_ability,r,side,slot)
    c=c._replace(trackers=_write(c.trackers,'shown_ability',side,p,ability,use))
    if original:
        c=c._replace(trackers=_write(c.trackers,'original_ability',side,p,ability,use&(borrowed==0)))
        c=_narrow(c,side,p,ability,use&(borrowed==0),'ability')
    else:c=c._replace(trackers=port.trace(c.trackers,c.port_tables,side,ability,use))
    return _refresh(c,side,p,use)


def _mon_fact(c,name,side,physical):
    """[B] one mon_features entry of `physical` on `side` (zeros when the carry lacks the key)."""
    t=c.trackers;r=j.arange(len(t.error))
    if t.mon_features is None or name not in t.mon_features:return j.zeros(len(r),j.int32)
    slot=j.clip(_li.take(t.physical_to_public,r,side,j.clip(physical,0,5)),0,5)
    return _li.take(t.mon_features[name],r,side,slot)


def reveal_item(c,state,side,item,use,*,original=True):
    p=_li.take(state.active,_r(state),side);use=_mask(use,len(c.trackers.error))
    c=c._replace(trackers=_write(c.trackers,'shown_item',side,p,item,use))
    if not original:return _refresh(c,side,p,use)
    # Fidelity M1: B2 narrows by public_items.matching_item, which is 0 for a mutated item (Trick, Knock Off)
    # while item_belief_narrowing is False (src/pokezero/observation.py:382): a later reveal of a Tricked-in
    # item (its Leftovers heal, its berry) never narrows the holder's belief.
    mutated=_mon_fact(c,'item_mutated',side,p)!=0
    c=_narrow(c,side,p,item,use&~mutated,'item')
    return _refresh(c,side,p,use&mutated)


def entry(c,before,after,side,table,use):
    c=observe(c,after,use)
    if table is None:return c
    ability=_get(before,'ability',side);starts=use&(_get(after,'hp',side)>0)
    # Only an actual shared line reveals. Gen3 Intimidate prints -ability only
    # with a living foe that has no Substitute (otherwise an owner-only hint);
    # a weather ability prints -weather only when setWeather changes the field
    # (the same permanent weather is a silent no-op): entry.py's change test.
    requested=table.weather_ability[ability]
    foe_open=(_get(before,'hp',1-side)>0)&(_get(before,'substitute_hp',1-side)==0)
    sets_weather=(requested!=0)&((before.weather!=requested)|(before.weather_counter>0))
    reveal=starts&(sets_weather|(table.intimidate[ability]&foe_open))
    c=reveal_ability(c,after,side,reveal,ability=ability)
    blocked=starts&table.intimidate[ability]&foe_open&table.attack_drop_immune[_get(before,'ability',1-side)]
    c=reveal_ability(c,before,1-side,blocked)
    changed=starts&(table.weather_ability[ability]!=0)&((before.weather!=after.weather)|(before.weather_counter!=after.weather_counter))
    c=c._replace(trackers=port.spikes(c.trackers,before,after,side,use,tr._percent))
    c=weather(c,before,after,changed,permanent=True,side=side)
    if table.trace is not None:
        copied=starts&table.trace[ability]&(_get(before,'hp',1-side)>0)
        c=reveal_ability(c,before,side,copied,ability=ability)
        c=reveal_ability(c,before,1-side,copied)
        p=_li.take(after.active,_r(after),side)
        c=c._replace(trackers=_write(c.trackers,'borrowed_ability',side,p,_get(after,'ability',side),copied))
        c=reveal_ability(c,after,side,copied,original=False)
    current=_get(after,'ability',side)
    cure=starts&(_get(before,'status',side)!=0)&(_get(after,'status',side)==0)
    if table.own_tempo is not None:
        cure |= starts&table.own_tempo[current]&(_get(before,'confusion_counter',side)>0)
        # Fidelity patch B: Own Tempo's onUpdate removes a Baton-passed confusion (-end|X|confusion).
        c=confusion_end(c,after,side,starts&table.own_tempo[current]&(_get(before,'confusion_counter',side)>0))
    c=reveal_ability(c,after,side,cure)
    # Pressure's Gen3 start line is owner-only; never shared with the foe here.
    return c


def depart(c,before,after,side,table,use):
    p=_li.take(before.active,_r(before),side)
    if table is not None:
        cure=use&(_get(before,'hp',side)>0)&(_get(before,'status',side)!=0)&table.natural_cure[_get(before,'ability',side)]
        c=reveal_ability(c,before,side,cure)
        c=observe(c,after,cure)
    # A borrowed ability ends silently; known original identity is not guessed.
    c=c._replace(trackers=_write(c.trackers,'shown_ability',side,p,_li.take(c.trackers.original_ability,_r(before),side,j.clip(_li.take(c.trackers.active_slot,_r(before),side),0,5)),use&(_li.take(c.trackers.borrowed_ability,_r(before),side,j.clip(_li.take(c.trackers.active_slot,_r(before),side),0,5))!=0)))
    c=c._replace(trackers=_write(c.trackers,'borrowed_ability',side,p,j.zeros_like(use,j.int32),use))
    c=type_revert(c,before,after,side,p,use)
    return _refresh(c,side,p,use&~_li.take(c.trackers.fainted,_r(before),side,j.clip(_li.take(c.trackers.active_slot,_r(before),side),0,5)))


def type_revert(c,before,after,side,physical,use):
    """A departing mon's public live-type overlay (Color Change) ends.

    B2's public_types overlay belongs to the seat's current encounter and every
    arrival clears it, so a benched mon shows its species base types (the team
    writer's), exactly what install writes. The write follows the actual engine
    departure revert (entry.depart: types <- base_types), the same operand
    color_change wrote; without species tables the overlay is cleared to zero.
    """
    r=_r(before);b=len(r);p=j.clip(physical,0,5)
    reverted=_mask(use,b)&j.any(_li.take(before.types,r,side,p)!=_li.take(after.types,r,side,p),axis=1)
    base=j.zeros((b,2),j.int32)
    if c.belief_tables is not None and hasattr(c.belief_tables,'species_types'):
        species=_li.take(before.species,r,side,p);tab=c.belief_tables.species_initial_mask
        known=(species>0)&(species<tab.shape[0])
        base=j.where(known[:,None],c.belief_tables.species_types[j.clip(species,0,tab.shape[0]-1)],0)
    return c._replace(trackers=_write(c.trackers,'shown_types',side,physical,base,reverted))


def move(c,state,side,move,use,*,pp_use=False,original=True,count_turn=True,preparation=False):
    """Actual declarations charge public PP, never hidden physical PP cost.

    The ledger for side s is read as opponent knowledge only by seat 1-s.
    That observer knows its OWN active ability, including owner-only Pressure;
    using it for PP accounting does not reveal that ability to the acting seat.
    Original source targeting applies; called Sleep Talk children stay uncharged.
    """
    # Gen3 emits cant(slp) even for Sleep Talk/Snore, then an actual usable
    # original MOVE classifies that cant as skippedTime. Called children and
    # preparation-only declarations must not consume it twice.
    r=_r(state);p=_li.take(state.active,r,side);slot=j.clip(_li.take(c.trackers.active_slot,r,side),0,5)
    facts=c.trackers.mon_features
    skipped=use&j.asarray(original)&~j.asarray(preparation)&(_li.take(facts['sleep_cant_pending'],r,side,slot)!=0)&(_li.take(facts['sleep_usable_pending'],r,side,slot)!=0)
    c=feature(c,side,p,'sleep_skipped_turns',j.ones_like(p),skipped,add=True)
    for key in ('sleep_cant_pending','sleep_usable_pending'):c=feature(c,side,p,key,j.zeros_like(p),skipped)
    actual_move=move;move_line=_mask(use,len(r))&(move>0)
    if c.belief_tables is not None and hasattr(c.belief_tables,'public_move'):
        valid=(move>=0)&(move<len(c.belief_tables.public_move))
        at=j.clip(move,0,len(c.belief_tables.public_move)-1)
        evidence=valid&c.belief_tables.source_move_evidence[at]
        move=j.where(valid,c.belief_tables.public_move[at],0)
    else:evidence=j.ones_like(use)
    t=c.trackers;r=_r(state);p=_li.take(state.active,r,side);slot=j.clip(_li.take(t.physical_to_public,r,side,p),0,5)
    old=_li.take(t.observed_moves,r,side,slot);has=j.any(old==move[:,None],axis=1);empty=j.argmax(old==0,axis=1)
    at=j.where(has,j.argmax(old==move[:,None],axis=1),empty)
    declared=use&(move>0)
    # Struggle is an ordinary MOVE header but is never a real-set reveal or
    # PP charge in the original public_moves.record_move writer.
    room=has|j.any(old==0,axis=1);use=declared&evidence&room
    previous=_li.take(t.observed_moves,r,side,slot,at)
    observed=_li.put(t.observed_moves,r,(side,slot,at),j.where(use,move,previous))
    count=_li.take(t.pp_used,r,side,slot,at)
    extra=j.zeros_like(use)
    if (c.belief_tables is not None and hasattr(c.belief_tables,'public_foe_targeted')
            and hasattr(c.belief_tables,'public_pressure')):
        targets=c.belief_tables.public_foe_targeted;pressure=c.belief_tables.public_pressure
        foe=1-side
        # This is the receiving observer's exact owning fact, not the acting
        # seat's hidden opponent fact. Keep all shared reveal arrays untouched.
        running=_get(state,'ability',foe)
        valid_ability=(running>0)&(running<len(pressure))
        own_pressure=valid_ability&pressure[j.clip(running,0,len(pressure)-1)]&(_get(state,'hp',foe)>0)
        target_known=(actual_move>0)&(actual_move<len(targets))
        extra=own_pressure&target_known&targets[j.clip(actual_move,0,len(targets)-1)]
    charge=(use&j.asarray(pp_use)).astype(j.int32)*(1+extra.astype(j.int32))
    pp=_li.put(t.pp_used,r,(side,slot,at),count+charge)
    c=c._replace(trackers=t._replace(observed_moves=observed,pp_used=pp))
    # The child [from] line and preparation-only -singleturn annotation can
    # reveal a move, but neither is an ordinary MOVE header/decision datum.
    header=count_turn
    if header:
        c=c._replace(trackers=_write(c.trackers,'last_moves',side,p,move,declared))
        c=c._replace(trackers=port.anchor(c.trackers,side,declared,loaf=False))
    if not preparation:
        c=feature(c,side,p,'moves_shown',j.ones_like(move),declared,add=True)
    if header and c.trackers.mon_features is not None and 'choice_locked' in c.trackers.mon_features:
        # Fidelity M1: B2's CHOICE_MOVE (public_active) on the holder's own move line (no [from]: not a
        # called child, not a preparation): a publicly received Choice Band now locks (column 40).
        public=_mon_fact(c,'choice_public',side,p)!=0
        c=feature(c,side,p,'choice_locked',j.ones_like(p),move_line&public&~j.asarray(preparation))
    if header:
        t=c.trackers;td=t.tendency;first=declared&~_li.take(td.moved,r,side)
        c=feature(c,side,p,'stayed_and_attacked',j.ones_like(move),first,add=True)
        c=matchup(c,side,first,1)
        t=c.trackers;td=t.tendency
        c=c._replace(trackers=t._replace(tendency=td._replace(moved=_li.put(td.moved,r,(side,),_li.take(td.moved,r,side)|declared))))
        c=opportunity(c,side,declared&(_get(state,'charge_move',side)==0))
    c=_narrow(c,side,p,move,declared&evidence,'move') if original else c
    return _refresh(c,side,p,declared&~j.asarray(preparation))


def item_changes(c,before,after,use):
    for side in range(2):
        old=_get(before,'item',side);consumed=use&(old>0)&(_get(after,'item',side)==0)
        c=reveal_item(c,before,side,old,consumed)
        c=c._replace(trackers=_write(c.trackers,'item_removed',side,_li.take(after.active,_r(after),side),j.ones_like(consumed),consumed))
        # Fidelity M1: the consumption's -enditem is B2's ITEM_END: the holder's Trick/Choice flags end.
        for name in STINT_ITEM_KEYS:
            if c.trackers.mon_features is not None and name in c.trackers.mon_features:
                p=_li.take(after.active,_r(after),side);c=feature(c,side,p,name,j.zeros_like(p),consumed)
        # Fidelity patch B: a Lum Berry's onEat removes confusion (-end|X|confusion), e.g. one Baton-passed in.
        # States without the leaf (authored fixtures) carry no confusion transition.
        # Fixture-only guard: transition.py and secondary.py callers always pass BattleState, which has the leaf.
        if hasattr(before,'confusion_counter') and hasattr(after,'confusion_counter'):
            c=confusion_end(c,after,side,_mask(use,len(c.trackers.error))&_confusion_ended(before,after,side))
    return observe(c,after,use)


def white_herb(c,before,after,side,use):
    """One White Herb use by `side`'s active (fidelity WH): -enditem|X|White Herb, then -clearnegativeboost|X|[silent]
    (data/items.ts whiteherb: useItem, then onUse).

    B2 folds them as a consumption, like a berry eaten: public_items CONSUME (the item revealed and removed; not a
    mutation, so the item ledger stays), public_active ITEM_END (the per-stint item flags end), and CLEAR_NEGATIVE on
    the holder's public stages. That is item_changes' consumption for this one active, with the holder's stage copy
    observe makes; the lines carry no HP or status, so neither active's condition is polled here.
    """
    b=len(c.trackers.error);use=_mask(use,b);r=_r(after);p=_li.take(after.active,r,side)
    c=reveal_item(c,before,side,_get(before,'item',side),use)
    c=c._replace(trackers=_write(c.trackers,'item_removed',side,p,j.ones_like(use),use))
    for name in STINT_ITEM_KEYS:
        if c.trackers.mon_features is not None and name in c.trackers.mon_features:
            c=feature(c,side,p,name,j.zeros_like(p),use)
    return c._replace(trackers=_write(c.trackers,'boosts',side,p,_get(after,'boosts',side),use))


def knock_off(c,before,after,side,use):
    """One actual Knock Off removal: -enditem|Y|Item|[from] move: Knock Off|[of] X.

    B2: public_items REMOVE (revealed = the removed item, removed, mutated), the
    public_active ITEM_END clear of the choice/swap flags, and public_damage's
    removal credit on the victim's side (items_removed[victim] += 1). The
    removed item is shown without narrowing: a mutated item no longer narrows
    (_reset_item_narrowing). Never routed through item_changes, which would
    treat it as a consumption and narrow by it.
    """
    b=len(c.trackers.error);use=_mask(use,b);r=_r(after);p=_li.take(after.active,r,side)
    c=c._replace(trackers=_write(c.trackers,'shown_item',side,p,_get(before,'item',side),use))
    c=c._replace(trackers=_write(c.trackers,'item_removed',side,p,j.ones_like(use),use))
    for name in STINT_ITEM_KEYS:
        if c.trackers.mon_features is not None and name in c.trackers.mon_features:
            c=feature(c,side,p,name,j.zeros_like(p),use)
    # Fidelity M1: B2's ledger REMOVE marks the item mutated and, for an unmutated mon with no recorded
    # original, records the removed item as its original assignment (public_items.record own_removal).
    if c.trackers.mon_features is not None and 'item_mutated' in c.trackers.mon_features:
        lost=_get(before,'item',side)
        fresh=(_mon_fact(c,'item_mutated',side,p)==0)&(_mon_fact(c,'item_original',side,p)==0)
        c=feature(c,side,p,'item_original',lost,use&fresh)
        c=feature(c,side,p,'item_mutated',j.ones_like(p),use)
    c=_reset_item_narrowing(c,side,p,use)
    t=c.trackers;fields=t.field_features
    if fields is not None and 'items_removed_credit' in fields:
        credit=fields['items_removed_credit']
        credit=_li.put(credit,r,(side,),_li.take(credit,r,side)+use.astype(credit.dtype))
        c=c._replace(trackers=t._replace(field_features={**fields,'items_removed_credit':credit}))
    return c


def trick(c,before,after,side,use,choice_items,herb_used=None):
    """One successful Trick by `side` (fidelity M1, design 2.5). Showdown prints -activate|S|move: Trick|[of] T,
    then for T and for S: -item|X|received|[from] move: Trick, or -enditem|X|lost|[silent]|[from] move: Trick
    when nothing comes back (data/moves.ts trick onHit).

    Per mon, as B2 folds those lines: the ledger (public_items.record TRICK_RECEIVE / REMOVE) reveals the
    received item, or the lost one when nothing came back (shown_item, read as the known item); removed
    is "nothing received" (item_removed); the item is mutated; and a mon unmutated with no recorded original
    before this Trick gets the item it held as its original (TRICK_RECEIVE's partner credit or REMOVE's own
    removal). The seat flags (public_active): -item is ITEM_SET (item_swapped, choice_public = a Choice item
    received, choice_locked reset), -enditem is ITEM_END (all three reset). A mutated item no longer narrows
    (design 13 item 4: B2's matching key is 0), so both mons' item narrowing is reset.
    B2's identical-swap quirk: a Band <-> Band Trick leaves the state unchanged but prints the same lines,
    so `use` is the transition's success mask, never a before/after item difference.
    Fidelity WH: herb_used [B,2] (by side) marks a receiver that used a received White Herb inside onHit; its
    -enditem and -clearnegativeboost precede its own -item line (the target's lines first, then the user's), and B2
    ends the Trick pairing at any non-item line (public_items.begin_event), so a partner credit read from a -item line
    after one is lost: the user's from the target's line when the target used it, the target's from the user's line
    when either did. A mon that received nothing keeps its own removal credit (its -enditem [silent]).
    """
    b=len(c.trackers.error);use=_mask(use,b);r=_r(after)
    if herb_used is None:herb_user=herb_target=j.zeros(b,bool)
    else:
        herb_used=j.asarray(herb_used,bool);herb_user=herb_used[r,side];herb_target=herb_used[r,1-side]
    for m,broken in ((side,herb_target),(1-side,herb_target|herb_user)):
        p=_li.take(after.active,r,m)
        received=_get(after,'item',m);lost=_get(before,'item',m);got=received>0
        c=c._replace(trackers=_write(c.trackers,'shown_item',m,p,j.where(got,received,lost),use))
        c=c._replace(trackers=_write(c.trackers,'item_removed',m,p,~got,use))
        facts=c.trackers.mon_features
        if facts is not None and 'item_mutated' in facts:
            fresh=(_mon_fact(c,'item_mutated',m,p)==0)&(_mon_fact(c,'item_original',m,p)==0)
            c=feature(c,m,p,'item_original',lost,use&fresh&~(broken&got))
            c=feature(c,m,p,'item_mutated',j.ones_like(p),use)
        choice=got&j.asarray(choice_items)[j.clip(received,0,len(choice_items)-1)]
        for name,value in (('item_swapped',got),('choice_public',choice),('choice_locked',j.zeros_like(got))):
            if c.trackers.mon_features is not None and name in c.trackers.mon_features:
                c=feature(c,m,p,name,value.astype(j.int32),use)
        c=_reset_item_narrowing(c,m,p,use)
    return c


def recharge_turn(c,state,side,use):
    """B2 on cant|X|recharge (fidelity M1): CLEAR_RECHARGE ends the seat's mustrecharge flag, STALL_BREAK
    its stall count (already zero after the Hyper Beam move), EXPIRE_ACTION its Destiny Bond (the
    destiny_bond hook at the same attempt). It is no decision opportunity (public_tendency drops recharge
    cants) and no Truant anchor. A recharge whose action a Gen 3 faint cancels expires silently at the
    residual and B2 keeps the flag: its stale bit, matched on purpose (design 13 item 3); arrival (install)
    and the faint clear end it."""
    use=_mask(use,len(c.trackers.error));p=_li.take(state.active,_r(state),side)
    c=volatile(c,state,side,'mustrecharge',j.zeros_like(use),use)
    return feature(c,side,p,'stall_counter',j.zeros_like(p),use)


def destiny_bond(c,state,side,use):
    """The active's public destinybond bag bit, written from the actual volatile.

    B2 adds it on -singlemove|X|Destiny Bond and expires it (EXPIRE_ACTION) at
    the holder's next move or cant line or confusion self-hit; the engine
    clears the bit at that same move attempt and sets it when Destiny Bond
    executes. FAINT and arrival clears are the existing faint_clear/install.
    """
    from .state import DESTINY_BOND
    held=(_get(state,'volatile',side)&j.uint32(DESTINY_BOND))!=0
    return volatile(c,state,side,'destinybond',held,_mask(use,len(c.trackers.error)))


def party_cure(c,before,after,side,use,table=None):
    # Heal Bell's actual living non-Soundproof write; known slots only. Never
    # reveal unseen bench identity or private status from a scan.
    if table is not None:c=reveal_ability(c,before,side,use&table.soundproof[_get(before,'ability',side)])
    r=_r(before)
    for p in range(6):
        changed=use&(_li.take(before.status,r,side,p)!=_li.take(after.status,r,side,p))
        old=c.trackers.status;slot=_li.take(c.trackers.physical_to_public,r,side,p);at=j.clip(slot,0,5)
        yes=changed&(slot>=0)
        c=c._replace(trackers=c.trackers._replace(status=_li.put(old,r,(side,at),j.where(yes,_li.take(after.status,r,side,p),_li.take(old,r,side,at)))))
        cured=yes&(_li.take(after.status,r,side,p)==0)
        c=c._replace(trackers=port.condition(c.trackers,side,j.full(len(r),p,j.int32),_li.take(before.status,r,side,p),_li.take(after.status,r,side,p),yes))
        for name in ('sleep_turns','rest_sleep','toxic_stage','sleep_skipped_turns','sleep_cant_pending','sleep_usable_pending'):
            c=feature(c,side,p,name,j.zeros(len(r),j.int32),cured)
        c=_refresh(c,side,p,yes)
    c=c._replace(trackers=port.team_cure(c.trackers,side,use))
    return c


def damage(c,state,side,dealt,taken,use):
    # Keep observer-relative fractions; only readout of the OWN victim uses
    # exact HP. Foe-victim public fractions come from shown percentage points.
    r=_r(state);foe=1-side;p=_li.take(state.active,r,foe);slot=j.clip(_li.take(c.trackers.physical_to_public,r,foe,p),0,5)
    previous=_li.take(c.trackers.hp_percent,r,foe,slot)
    current=tr._percent(_get(state,'hp',foe),_get(state,'max_hp',foe))
    shown=j.maximum(0,previous-current).astype(j.float32)/100.0
    exact=taken.astype(j.float32)/j.maximum(_get(state,'max_hp',foe),1).astype(j.float32)
    if c.trackers.damage_history is not None:
        return c._replace(trackers=c.trackers._replace(damage_history=damage_history.direct(
            c.trackers.damage_history,side,shown,exact,use & (taken>0))))
    c=c._replace(trackers=_write(c.trackers,'damage_exact_dealt',side,_li.take(state.active,r,side),exact,use))
    c=c._replace(trackers=_write(c.trackers,'damage_exact_taken',foe,p,exact,use))
    c=feature(c,side,_li.take(state.active,r,side),'last_damage_dealt',shown,use)
    return feature(c,foe,p,'last_damage_taken',shown,use)


def negative_hp_line(c,state,side,use,*,source_class):
    # Called before observe at each actual HP-line cutpoint. The shared hp_line
    # hook composes this notification with observer-relative damage history.
    return c._replace(trackers=negative_items.hp_line(c.trackers,state,side,use,source_class=source_class))


FAINT_CLEAR_KEYS=('confusion_turns','encore_turns','wrap_trap_turns','stall_counter','sub_present',
                  'item_swapped','choice_locked','choice_public')


def faint_clear(c,before,after,use):
    """B2's FAINT public clear for an active whose HP reached zero this step.

    public_active.apply(FAINT) empties the fainted seat's volatile bag (recharge
    included), resets its confusion/encore/partial-trap clocks, stall counter
    and choice/swap flags, and clears Mean Look on BOTH seats. The fainted mon
    stays the active token until its replacement arrives, so the replacement
    request must see this. The SUB fraction is the bag's substitute bit in B2.
    Keys absent from this carry (deferred features) are skipped.
    """
    r=_r(after);b=len(r);use=_mask(use,b);hit=[]
    for side in range(2):
        p=_li.take(after.active,r,side)
        hit.append(use&(_li.take(before.hp,r,side,j.clip(p,0,5))>0)&(_get(after,'hp',side)==0))
    for side in range(2):
        p=_li.take(after.active,r,side)
        c=c._replace(trackers=_write(c.trackers,'public_volatiles',side,p,j.zeros((b,len(VOLATILE_NAMES)),bool),hit[side]))
        for name in FAINT_CLEAR_KEYS:
            if name in c.trackers.mon_features:c=feature(c,side,p,name,j.zeros_like(p),hit[side])
        c=feature(c,side,p,'meanlook_trap',j.zeros_like(p),hit[0]|hit[1])
    return c


def turn_end(c,before,after,ordinary,*,upkeep=None,leftovers=None):
    c=observe(c,after,~before.done)
    c=faint_clear(c,before,after,~before.done)
    c=derived_volatiles(c,after,~before.done)
    for side in range(2):
        p=_li.take(after.active,_r(after),side);alive=ordinary&(_get(after,'hp',side)>0)
        clocks=[('confusion_turns',_get(after,'confusion_counter',side)>0),('encore_turns',_get(after,'encore_counter',side)>0)]
        # Fidelity M2: WRAP_TRAP_TURNS follows the same TURN rule while partiallytrapped is held.
        if hasattr(after,'partial_trap_counter') and 'wrap_trap_turns' in c.trackers.mon_features:
            clocks.append(('wrap_trap_turns',_get(after,'partial_trap_counter',side)>0))
        for name,active in clocks:
            c=feature(c,side,p,name,j.ones_like(p),alive&active,add=True)
            c=feature(c,side,p,name,j.zeros_like(p),ordinary&~active)
    t=c.trackers;fields=t.field_features
    up=fields['weather_upkeeps']+(ordinary&(after.weather!=0)).astype(j.int32)
    c=c._replace(trackers=t._replace(field_features={**fields,'weather_upkeeps':up}))
    if upkeep is not None:
        facts=c.trackers.mon_features
        pending=facts['sleep_cant_pending']!=0
        on=_mask(upkeep,len(c.trackers.error))[:,None,None]
        c=c._replace(trackers=c.trackers._replace(mon_features={**facts,
            'sleep_skipped_turns':j.where(on&pending,0,facts['sleep_skipped_turns']),
            'sleep_cant_pending':j.where(on,0,facts['sleep_cant_pending']),
            'sleep_usable_pending':j.where(on,0,facts['sleep_usable_pending'])}))
        c=c._replace(trackers=port.upkeep(c.trackers,upkeep))
        c=c._replace(trackers=negative_items.upkeep(c.trackers,c.belief_tables,leftovers,upkeep,current_state=after))
    # A paused action/residual or forced replacement has no next TURN line.
    # A completed replacement may open that line even without residual work.
    ready=(~before.done&~after.done&~after.pending_residual&~j.any(after.forced_switch,axis=1)
        &(c.trackers.turn<after.turn+1))
    return public_turn(c,after,ready,after.turn+1)


def color_change(c,before,after,side,use):
    changed=use&j.any(_get(before,'types',side)!=_get(after,'types',side),axis=1)
    c=reveal_ability(c,after,side,changed)
    p=_li.take(after.active,_r(after),side)
    return c._replace(trackers=_write(c.trackers,'shown_types',side,p,_get(after,'types',side),changed))


def drag(c,before,after,side,table,use):
    # One actual drag operation owns exactly one depart/install/entry. Its
    # pre-entry incoming private operands already exist in the actual before
    # state; only public handler results are exposed.
    c=depart(c,before,after,side,table,use)
    # The dragged-out mon's slot drops its stages, as withdraw does for a switch (entry.depart zeroes the engine's).
    # The fainted-active carry (engine_device_trackers._observe) keeps a slot's stages at hp 0, so a stale bench carry
    # would show again if a later drag returned this mon into lethal Spikes, where B2's drag arrival reads 0
    # (public_mon_state.py:111-112).
    c=c._replace(trackers=_write(c.trackers,'boosts',side,_li.take(before.active,_r(before),side),
        j.zeros((len(c.trackers.error),7),j.int32),use))
    c=install(c,after,side,use)
    incoming=before._replace(active=_li.put(before.active,_r(before),(side,),_li.take(after.active,_r(after),side)))
    return entry(c,incoming,after,side,table,use) if table is not None else observe(c,after,use)


def public_move_state(c,before,after,side,move,use):
    """Observable volatile starts/ends, not private remaining durations."""
    for target in (side,1-side):
        p=_li.take(after.active,_r(after),target)
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
        # Fidelity M1: -mustrecharge (mustrecharge onStart) is B2's RECHARGE: set when the counter starts (0 -> 2).
        # Never cleared here: B2 clears it only on cant|X|recharge (recharge_turn), arrival or a faint.
        if hasattr(before,'recharge_counter') and hasattr(after,'recharge_counter'):
            started=use&(_get(before,'recharge_counter',target)==0)&(_get(after,'recharge_counter',target)>0)
            c=volatile(c,after,target,'mustrecharge',j.ones_like(started),started)
        # Fidelity M2: a partial trap ended by the move (Rapid Spin's removeVolatile, a Substitute's onStart) prints
        # -end|X|Wrap|[partiallytrapped]: B2's WRAP_END resets the clock at once.
        if (hasattr(before,'partial_trap_counter') and hasattr(after,'partial_trap_counter')
                and 'wrap_trap_turns' in c.trackers.mon_features):
            ended=use&(_li.take(before.active,_r(before),target)==_li.take(after.active,_r(after),target))&(
                _get(before,'partial_trap_counter',target)>0)&(_get(after,'partial_trap_counter',target)==0)
            c=feature(c,target,_li.take(after.active,_r(after),target),'wrap_trap_turns',j.zeros_like(ended,j.int32),ended)
    c=derived_volatiles(c,after,use)
    return observe(c,after,use)


PERISH_BAG=('perish0','perish1','perish2','perish3')


def derived_volatiles(c,state,use):
    """Fidelity M2 (design 2.5): the actives' perish, yawn, partiallytrapped, encore and flashfire bag bits, from the actual
    state.

    B2 (public_active.project_line) folds -start|X|perishN (PERISH_START: perishN replaces the other perish bits;
    perish3 [silent] at Perish Song, then perishN each residual), -start|X|move: Yawn / -end|X|move: Yawn|[silent],
    and -activate|X|move: Wrap / -end|X|Wrap|[partiallytrapped] (silent or not). Each line comes exactly when the
    engine counter starts, changes or ends, so at every boundary B2's bag is perish{min(counter,3)} (perish0 only
    precedes the faint, whose FAINT clear empties the bag), yawn = counter>0 and partiallytrapped = counter>0.
    A switch-out drops them (withdraw), a Baton target carries perish and partiallytrapped (baton_copy). States
    without the leaves (authored fixtures) are left alone.

    Encore (the device harness's OPEN BUG C): B2 folds -start|X|Encore and -end|X|Encore (Showdown's encore condition
    onEnd prints it both at the residual expiry and when the encored move's PP runs out; gen3 residual order 10,
    subOrder 14), and an Encore's silent ends (switch-out, faint) clear the bag through withdraw and the faint clear.
    The engine's encore_counter starts with the -start and reaches 0 exactly at the -end, so encore = counter>0 is B2's
    bag at every boundary. public_move_state's write at the Encore move stays (it agrees); this is what clears it at
    the residual, where transition.py's Encore block calls no hook. The engine zeroes a fainted holder's encore_counter
    only at the step's end (transition.py's faint clear, after the residuals), so a read between the faint and that
    clear (a public_move_state later in the step, a fixture) sees a live counter on a fainted holder: hence the hp gate,
    as flashfire's. B2's FAINT clear has emptied the bag at the faint.

    Flash Fire (the device harness's OPEN BUG E): Showdown's Flash Fire onTryHit prints -start|X|ability: Flash Fire
    when it adds the flashfire volatile, and B2 folds it into the bag's flashfire entry; a Fire move into a holder that
    already has it adds nothing (-immune, no new -start), and the volatile ends silently: noCopy, so a Baton Pass drops
    it (B2's carry set has no flashfire, baton_copy's neither), a switch-out (withdraw) and a faint (B2's FAINT clear).
    The engine bit absorption.FLASH_FIRE in state.volatile is set by absorption.try_hit exactly then, kept through a
    second activation, cleared by a departure and not copied by baton.copy_to_recipient (LEECH_SEED only); a faint
    does not clear it, so flashfire = bit & hp>0 is B2's bag at every boundary."""
    if not hasattr(state,'perish_counter') or c.trackers.public_volatiles is None:return c
    use=_mask(use,len(c.trackers.error));r=_r(state)
    for side in range(2):
        p=_li.take(state.active,r,side);slot=j.clip(_li.take(c.trackers.physical_to_public,r,side,p),0,5)
        bag=_li.take(c.trackers.public_volatiles,r,side,slot)
        perish=_get(state,'perish_counter',side);shown=j.minimum(perish,3)
        for k,name in enumerate(PERISH_BAG):bag=bag.at[:,VOLATILE_NAMES.index(name)].set((perish>0)&(shown==k))
        bag=bag.at[:,VOLATILE_NAMES.index('yawn')].set(_get(state,'yawn_counter',side)>0)
        bag=bag.at[:,VOLATILE_NAMES.index('partiallytrapped')].set(_get(state,'partial_trap_counter',side)>0)
        bag=bag.at[:,VOLATILE_NAMES.index('encore')].set((_get(state,'encore_counter',side)>0)&(_get(state,'hp',side)>0))
        bag=bag.at[:,VOLATILE_NAMES.index('flashfire')].set(
            ((_get(state,'volatile',side)&j.uint32(FLASH_FIRE))!=0)&(_get(state,'hp',side)>0))
        c=c._replace(trackers=_write(c.trackers,'public_volatiles',side,p,bag,use))
    return c


def weather(c,before,after,use,*,permanent=False,side=None):
    t=c.trackers;fields=t.field_features
    changed=use&((before.weather!=after.weather)|(before.weather_counter!=after.weather_counter))
    if side is not None:c=c._replace(trackers=port.weather(c.trackers,side,after.weather,changed,ability=permanent))
    c=c._replace(trackers=c.trackers._replace(field_features={**fields,
        'weather_upkeeps':j.where(changed,0,fields['weather_upkeeps']),
        'weather_permanent':j.where(changed,j.asarray(permanent)&(after.weather!=0),fields['weather_permanent'])}))
    return observe(c,after,use)


def wish(c,side,use,*,start=False,resolved=False):
    """Public declaration/actual heal only; silent expiry uses public TURN.

    The original parser stores the declaration turn, not the numerical engine
    counter or elapsed residual ticks. A pending second declaration cannot
    rearm. A full-HP landing has no heal line and stays pending until TURN.
    """
    t=c.trackers;f=t.field_features;r=j.arange(len(t.error));use=_mask(use,len(r))
    old=_li.take(f['wish_set_turn'],r,side);arm=(old<0)|((t.turn-old)>1)
    declared=j.where(use&j.asarray(start)&arm,t.turn,old)
    declared=j.where(use&~j.asarray(start)&_mask(resolved,len(r)),-1,declared)
    elapsed=t.turn-declared
    pending=(declared>=0)&(elapsed<=1)
    remaining=j.where((declared>=0)&(elapsed>=0)&(elapsed<=1),2-elapsed,0)
    return c._replace(trackers=t._replace(field_features={**f,
        'wish_set_turn':_li.put(f['wish_set_turn'],r,(side,),declared),
        'wish_turns':_li.put(f['wish_turns'],r,(side,),j.where(use,remaining,_li.take(f['wish_turns'],r,side))),
        'wish_pending':_li.put(f['wish_pending'],r,(side,),j.where(use,pending,_li.take(f['wish_pending'],r,side)))}))


def before_move(c,state,side,result,use,*,loafing=False,sleep_usable=False):
    p=_li.take(state.active,_r(state),side)
    truant_cant=use&j.asarray(loafing)&((result.status!=5)|j.asarray(sleep_usable))&(result.status!=6)
    c=c._replace(trackers=port.anchor(c.trackers,side,truant_cant,loaf=True))
    asleep=use&(_get(state,'status',side)==5)&(result.status==5)
    c=feature(c,side,p,'sleep_turns',j.ones_like(p),asleep,add=True)
    c=feature(c,side,p,'sleep_cant_pending',j.ones_like(p),asleep)
    c=feature(c,side,p,'sleep_usable_pending',j.asarray(sleep_usable,j.int32),asleep)
    c=opportunity(c,side,use&~result.can_act&~result.confusion_self_hit)
    # Fidelity patch B (design 2.5): the snap-out (confusion's time reaches 0 at BeforeMove) prints
    # -end|X|confusion; public_move_state cannot see it, its before being the post-BeforeMove state.
    # A result without the field (an authored fixture) carries no confusion transition.
    # Fixture-only guard: transition.py always passes mechanics.BeforeMove, which has the field.
    left=getattr(result,'confusion_counter',None)
    if left is not None:c=confusion_end(c,state,side,use&(_get(state,'confusion_counter',side)>0)&(j.asarray(left)==0))
    return _refresh(c,side,p,use&(_get(state,'status',side)==5)&(result.status==5))


def volatile(c,state,side,name,value,use):
    r=_r(state);p=_li.take(state.active,r,side);slot=j.clip(_li.take(c.trackers.physical_to_public,r,side,p),0,5)
    old=_li.take(c.trackers.public_volatiles,r,side,slot);new=old.at[:,VOLATILE_NAMES.index(name)].set(value)
    return c._replace(trackers=_write(c.trackers,'public_volatiles',side,p,new,use))


def _confusion_ended(before,after,side):
    """[B] the same living active's actual confusion ended between before and after."""
    r=_r(after)
    return ((_li.take(before.active,r,side)==_li.take(after.active,r,side))&(_get(before,'confusion_counter',side)>0)
        &(_get(after,'confusion_counter',side)==0)&(_get(after,'hp',side)>0))


def confusion_end(c,state,side,use):
    """Fidelity patch B: B2's -end|X|confusion on the active of `side`.

    Showdown prints it whenever removeVolatile('confusion') ends a living active's
    confusion: the BeforeMove snap-out, a Lum Berry's onEat, Own Tempo's onUpdate
    (data/mods/gen4/conditions.ts, data/conditions.ts confusion onEnd). B2's REMOVE
    (public_active.apply) drops the bag entry and resets the CONFUSION_TURNS clock.
    A switch-out or a faint ends it silently: install/withdraw and faint_clear.
    """
    use=_mask(use,len(c.trackers.error));p=_li.take(state.active,_r(state),side)
    c=volatile(c,state,side,'confusion',j.zeros_like(use),use)
    return feature(c,side,p,'confusion_turns',j.zeros_like(p),use)


def baton_copy(c,state,side,outgoing,use):
    r=_r(state);p=_li.take(state.active,r,side);src=j.clip(_li.take(c.trackers.physical_to_public,r,side,outgoing),0,5)
    # Fidelity M2: perishN and partiallytrapped are Baton-transferred in B2 (src/pokezero/showdown.py
    # _BATON_PASS_TRANSFERRED_VOLATILES); yawn is noCopy and is not.
    names={'confusion','leechseed','substitute','solarbeam','partiallytrapped',*PERISH_BAG}
    allowed=j.asarray([name in names for name in VOLATILE_NAMES])
    bag=_li.take(c.trackers.public_volatiles,r,side,src)&allowed[None,:]
    c=c._replace(trackers=_write(c.trackers,'public_volatiles',side,p,bag,use))
    # Not meanlook_trap: B2 clears it at every arrival, Baton included (see install).
    for name in ('confusion_turns','sub_present','wrap_trap_turns'):
        if name not in c.trackers.mon_features:continue
        value=_li.take(c.trackers.mon_features[name],r,side,src)
        c=feature(c,side,p,name,value,use)
    c=c._replace(trackers=_write(c.trackers,'last_moves',side,p,j.full_like(p,-2),use))
    return c


def withdraw(c,state,side,physical,use):
    b=len(c.trackers.error)
    c=c._replace(trackers=_write(c.trackers,'boosts',side,physical,j.zeros((b,7),j.int32),use))
    c=c._replace(trackers=_write(c.trackers,'public_volatiles',side,physical,j.zeros((b,len(VOLATILE_NAMES)),bool),use))
    for name in ('turns_active','confusion_turns','encore_turns','stall_counter','sub_present','meanlook_trap','wrap_trap_turns'):
        if name in c.trackers.mon_features:c=feature(c,side,physical,name,j.zeros(b,j.int32),use)
    return c


def port_count(c,side,use,name):
    return c._replace(trackers=port.count(c.trackers,side,use,name))


def port_protect(c,side,acts,accuracy_pass,protected,protectable):
    return c._replace(trackers=port.protect(c.trackers,side,acts,accuracy_pass,protected,protectable))
