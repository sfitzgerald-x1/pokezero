"""Declared actual-kernel cutpoint risks, NOT source/consumer completeness.

First proposed domain is fresh construction + weather-table actions and their
already registered weather/Leftovers transports. Selection reads ORIGINAL
immutable kernel columns, never a move-name roster or original simulator hooks.
Other literal branch observations stick even after HP/effects revert.

The optional actual kernel callers record these declared cutpoints. Compilation
binds identity ONLY. The owning constructor/Core must retain this SAME carry
and compose genuine accounting and BOTH-owner drains. Zero risk,
all observed bits or a copied carry cannot establish exhaustive source coverage.
There is deliberately no complete/admitted/exhaustive/qualification boolean API.
"""
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
from pathlib import Path
from typing import NamedTuple
import weakref
import jax
import jax.numpy as jnp
import numpy as np
from . import engine_public_events as public, engine_execution_completeness as accounting
from . import public_mon_state, mechanics
from .state import BattleState


class Branch(IntEnum):
    CONSTRUCTOR=1
    PRE_MOVE=2
    SELECTED_MOVE=3
    HIT_OUTCOME=4
    FRAME_END=5
    SUBSTITUTE=6
    SELF_HEAL=7
    PAIN_SPLIT=8
    BELLY_DRUM=9
    SELF_BOOST=10
    HIT_SPA_DROP=11
    SECONDARY_STATS=12
    SECONDARY_FLINCH=13
    SEED_START=14
    WISH_START=15
    SPIN_CLEAR=16
    LINK_TRAP=17
    ENCORE=18
    SPIKES=19
    HAZE=20
    SELF_DESTRUCT=21
    RECOIL=22
    CONFUSION_HP=23
    ENTRY_CONFUSION_CURE=24


NON_WEATHER, SELF_EFFECT, BOOST, FIELD_VOLATILE, TARGET_OUTCOME, SELF_HP, BLOCKED_ACTION, TERMINAL, CONSTRUCTOR_STATE, NAMESPACE, ORDER, OVERFLOW, UPSTREAM = (1<<n for n in range(13))
SCHEMA='pokezero.actual_kernel.frame_risk.declared.v1'
UNKNOWN_KERNEL_RISK=1024
_COMPILED=weakref.WeakKeyDictionary()


class Tables(NamedTuple):
    binding: object
    engine_binding: object
    source_binding: object
    move_binding: object
    producer_binding: object
    accounting_binding: object
    selected_risk: object
    branch_risk: object
    direct_hit_binding: object       # GLOBAL8; zero means weather-only mode
    direct_candidates: object        # [M] immutable original table capability
    substitute_binding: object      # GLOBAL8, original selected self-write helper
    substitute_candidates: object   # [M], prospective only; actual Site11 REQUIRED


@dataclass(frozen=True,eq=False)
class Compiled:
    device: Tables


class Carry(NamedTuple):
    binding: object                    # GLOBAL8, including B==8
    accounting_binding: object         # GLOBAL8
    stream_id: object                  # all remaining fields lead with B
    frame: object
    opened: object
    risk: object
    observed: object                   # bitset of declared cutpoints CALLED
    counts: object                     # [B,24], not public-event counts


class Context(NamedTuple):
    """PRIVATE execution.Context.risks carry; owning Core retains it explicitly."""
    carry: Carry
    tables: Tables


class Outcome(NamedTuple):
    power: object
    critical: object
    hit: object
    immune: object
    numerator: object
    denominator: object
    absorbed: object
    substitute: object
    protected: object
    enduring: object
    recoil: object
    explodes: object


class CapturedDirectHit(NamedTuple):
    """PRIVATE SAME-call source values, not caller attestation or credit.

    The actual kernel passes its original registered context, fresh Site10
    ticket and literal write. The owning driver must establish frame origin;
    hot device arrays/copies cannot attest that origin on their own.
    """
    context: object
    ticket: object
    actor: object
    actor_physical: object
    target_physical: object
    move: object
    write: object


def _host():
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):raise ValueError('checked kernel risk boundary cannot be traced')


def compile_risks(engine_catalog,accounting_compiled,*,direct_hit_compiled=None,substitute_compiled=None):
    _host();record=accounting._checked(accounting_compiled)
    if record[2].causal.projector.catalog is not engine_catalog:raise ValueError('same original accounting/kernel catalog required')
    public.check_catalog(engine_catalog,engine_catalog.engine_table,expected_binding=engine_catalog.binding,
        expected_engine_binding=np.asarray(engine_catalog.engine_binding,np.uint32))
    table=engine_catalog.engine_table;m=len(table.power);risk=np.zeros(m,np.int32)
    weather=np.asarray(table.weather.move_weather) if table.weather is not None else np.zeros(m,np.int32)
    direct_binding=np.zeros(8,np.uint32);candidate=np.zeros(m,bool)
    if direct_hit_compiled is not None:
        from . import engine_direct_hit_controls as direct
        direct_data=direct.check_compiled(direct_hit_compiled)
        if direct._COMPILED[direct_hit_compiled][2] is not engine_catalog:
            raise ValueError('same original direct-hit/kernel catalog required')
        if int(direct_data.move_count)!=m:raise ValueError('exact direct-hit move namespace required')
        direct_binding=np.asarray(direct_data.binding,np.uint32)
        # Candidate only: every actual positive outcome additionally requires
        # its SAME captured Site10 span. Other static/attempt risk bits remain.
        positive=np.asarray(table.power)>0
        if table.variable_moves is not None:
            # These original resolver families have positive power even when
            # the source move's static base power is zero. Actual same-call
            # direct-hit tickets and all other risk guards still apply.
            for name in ('returns','frustration'):
                variable=np.asarray(getattr(table.variable_moves,name))
                if variable.shape!=(m,):raise ValueError('exact selected variable move columns required')
                positive |= variable
        candidate=positive&(np.asarray(engine_catalog.ids)>0)&(np.asarray(engine_catalog.targets)==1)
    substitute_binding=np.zeros(8,np.uint32);substitute_candidate=np.zeros(m,bool)
    if substitute_compiled is not None:
        from . import engine_substitute_callbacks as substitute
        data=substitute.check_compiled(substitute_compiled)
        if substitute._COMPILED[substitute_compiled][2] is not engine_catalog:
            raise ValueError('same original Substitute/kernel catalog required')
        move=int(data.move_id)
        if not 0<move<m:raise ValueError('canonical Substitute move bounds required')
        substitute_binding=np.asarray(data.binding,np.uint32);substitute_candidate[move]=True
    risk[(weather==0)&~candidate&~substitute_candidate]|=NON_WEATHER
    def flag(module,name,bit):
        if module is None:return
        values=np.asarray(getattr(module,name))
        if values.shape[:1]!=(m,):raise ValueError('exact selected kernel move columns required')
        on=np.any(values!=0,axis=tuple(range(1,values.ndim))) if values.ndim>1 else values!=0
        risk[on]|=bit
    for name in ('heal_half','weather_heal','rest','pain_split','belly_drum'):flag(table.effects,name,SELF_EFFECT)
    if table.effects is not None:risk[np.asarray(table.effects.substitute)&~substitute_candidate]|=SELF_EFFECT
    for name in ('self_boosts','hit_spa_drop','haze'):flag(table.effects,name,BOOST)
    for name in ('protect','endure'):flag(table.effects,name,TARGET_OUTCOME)
    for name in ('heal_bell','status'):flag(table.effects,name,FIELD_VOLATILE)
    for name in ('seed','wish','spin'):flag(table.sustain,name,FIELD_VOLATILE)
    flag(table.entry,'spikes_move',FIELD_VOLATILE)
    flag(table.linked_traps,'move',FIELD_VOLATILE);flag(table.encore,'move',FIELD_VOLATILE)
    if table.secondary is not None:
        chance=np.asarray(table.secondary.chance)
        if chance.shape!=(m,):raise ValueError('exact selected kernel move columns required')
        # Registered direct hits already visit the original secondary guards.
        # A possible proc is not an actual write: status/reflection callbacks
        # and stats/flinch branch guards retain their existing sticky refusals.
        risk[(chance!=0)&~candidate]|=FIELD_VOLATILE
    flag(table.self_destruct,'move',SELF_HP);flag(table.damage_moves,'recoil_den',SELF_HP)
    flag(table.solar_beam,'move',BLOCKED_ACTION);flag(table.focus_punch,'move',BLOCKED_ACTION)
    branch=np.zeros(len(Branch)+1,np.int32)
    for code in Branch:
        if 6<=int(code)<=9:branch[int(code)]=SELF_EFFECT
        elif 10<=int(code)<=12 or code==Branch.HAZE:branch[int(code)]=BOOST
        elif 13<=int(code)<=19 or code==Branch.ENTRY_CONFUSION_CURE:branch[int(code)]=FIELD_VOLATILE
        elif 21<=int(code)<=23:branch[int(code)]=SELF_HP
    files=('transition.py','secondary.py','entry.py','effects.py','absorption.py','sustain.py',
        'linked_traps.py','encore.py','mechanics.py','engine_execution_completeness.py','engine_kernel_frame_risk.py')
    inventory=dict(schema=SCHEMA,declared={x.name:int(x) for x in Branch},
        source_files={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in files},
        scope='DECLARED_CUTPOINT_IDENTITY_ONLY_UNWIRED_NO_COMPLETENESS',
        kernel=engine_catalog.binding,accounting=np.asarray(record[0].inventory_binding).tolist(),
        selected_risk=risk.tolist(),branch_risk=branch.tolist(),direct_hit_binding=direct_binding.tolist(),
        direct_candidates=candidate.tolist(),substitute_binding=substitute_binding.tolist(),
        substitute_candidates=substitute_candidate.tolist())
    raw=json.dumps(inventory,sort_keys=True,separators=(',',':'))
    words=lambda x:jnp.asarray(x,jnp.uint32)
    device=Tables(words(np.frombuffer(hashlib.sha256(raw.encode()).digest(),dtype='>u4').astype(np.uint32)),
        words(engine_catalog.engine_binding),words(engine_catalog.source_binding),words(engine_catalog.move_binding),
        record[0].producer_binding,record[0].inventory_binding,jnp.asarray(risk),jnp.asarray(branch),words(direct_binding),jnp.asarray(candidate),words(substitute_binding),jnp.asarray(substitute_candidate))
    result=Compiled(device);_COMPILED[result]=(device,tuple(device),engine_catalog,accounting_compiled,raw,direct_hit_compiled,substitute_compiled)
    return result


def check_compiled(compiled):
    _host();r=_COMPILED.get(compiled) if type(compiled) is Compiled else None
    if (r is None or compiled.device is not r[0] or any(a is not b for a,b in zip(compiled.device,r[1]))
            or any(not isinstance(a,jax.Array) or isinstance(a,jax.core.Tracer) for a in compiled.device)):
        raise ValueError('original immutable kernel risk compiler required')
    catalog=r[2];public.check_catalog(catalog,catalog.engine_table,expected_binding=catalog.binding,
        expected_engine_binding=np.asarray(catalog.engine_binding,np.uint32))
    accounting._checked(r[3])
    if r[5] is not None:
        from . import engine_direct_hit_controls as direct
        direct.check_compiled(r[5])
    if r[6] is not None:
        from . import engine_substitute_callbacks as substitute
        substitute.check_compiled(r[6])
    return compiled.device


def inventory_json(compiled):
    _host();check_compiled(compiled);return _COMPILED[compiled][4]


def _vector(value,b,dtype=jnp.int32):
    if dtype==jnp.int32:value=public_mon_state._int32(value)
    elif type(value) is bool:value=jnp.asarray(value)
    if getattr(value,'dtype',None)!=jnp.dtype(dtype) or getattr(value,'shape',None) not in ((),(b,)):
        raise ValueError('typed scalar/B actual kernel observation required')
    return jnp.broadcast_to(value,(b,))


def initialize_device(streams,tables):
    accounting._shape(streams,streams.shape,jnp.int32)
    if streams.ndim!=1 or not len(streams):raise ValueError('nonempty physical streams required')
    b=len(streams);ids=jnp.sort(streams);bad=(streams<=0)|jnp.any(ids[1:]==ids[:-1])
    return Carry(tables.binding,tables.accounting_binding,streams,jnp.zeros(b,jnp.int32),jnp.zeros(b,bool),
        jnp.where(bad,NAMESPACE,0),jnp.zeros(b,jnp.int32),jnp.zeros((b,len(Branch)),jnp.int32))


def _bound(carry,tape,tables):
    if type(carry) is not Carry or type(tables) is not Tables:raise ValueError('typed kernel risk carry/tables required')
    b=len(carry.stream_id)
    for x in (carry.binding,carry.accounting_binding,*tables[:6]):accounting._shape(x,(8,),jnp.uint32)
    for name in ('stream_id','frame','risk','observed'):accounting._shape(getattr(carry,name),(b,),jnp.int32)
    accounting._shape(carry.opened,(b,),jnp.bool_);accounting._shape(carry.counts,(b,len(Branch)),jnp.int32)
    accounting._shape(tables.selected_risk,(len(tables.selected_risk),),jnp.int32)
    accounting._shape(tables.branch_risk,(len(Branch)+1,),jnp.int32)
    accounting._shape(tables.substitute_binding,(8,),jnp.uint32)
    accounting._shape(tables.substitute_candidates,(len(tables.selected_risk),),jnp.bool_)
    accounting._shape(tables.direct_hit_binding,(8,),jnp.uint32)
    accounting._shape(tables.direct_candidates,(len(tables.selected_risk),),jnp.bool_)
    context=accounting._context(tape)
    if context is None:return jnp.zeros(b,bool),None
    if type(context.tables) is not accounting.Tables:raise ValueError('typed actual accounting context required')
    ledger=context.ledger
    same=(jnp.all(carry.binding==tables.binding)&jnp.all(carry.accounting_binding==tables.accounting_binding)
        &(carry.stream_id==tape.stream_id)&(carry.stream_id==ledger.stream_id)
        &jnp.all(ledger.inventory_binding==tables.accounting_binding)&(ledger.frame>0)&ledger.opened
        &(carry.stream_id>0)&(carry.frame>=0)&(carry.risk>=0)&(carry.observed>=0)
        &jnp.all(carry.counts>=0,axis=1))
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        same &= jnp.all(getattr(tape,name)==getattr(tables,name))
    for name in ('engine_binding','source_binding','producer_binding'):
        same &= jnp.all(getattr(context.tables,name)==getattr(tables,name))
    same &= jnp.all(context.tables.inventory_binding==tables.accounting_binding)
    return same,ledger


def begin_frame_device(carry,tape,tables,*,active):
    same,ledger=_bound(carry,tape,tables);b=len(carry.stream_id);active=_vector(active,b,jnp.bool_)
    frame=jnp.zeros(b,jnp.int32) if ledger is None else ledger.frame
    good=same&(carry.frame<2**31-1)&(frame==carry.frame+1)
    use=active&good
    return carry._replace(frame=jnp.where(use,frame,carry.frame),opened=jnp.where(active,good,carry.opened),
        observed=jnp.where(use,0,carry.observed),counts=jnp.where(use[:,None],0,carry.counts),
        risk=carry.risk|jnp.where(active&~good,ORDER|NAMESPACE,0))


def _observe(carry,tape,tables,code,value,*,active):
    same,ledger=_bound(carry,tape,tables);b=len(carry.stream_id);active=_vector(active,b,jnp.bool_)
    frame=jnp.zeros(b,jnp.int32) if ledger is None else ledger.frame
    same &= carry.opened&(carry.frame==frame)
    slot=int(code)-1;overflow=carry.counts[:,slot]>=2**31-1;use=active&same&~overflow
    count=jnp.minimum(carry.counts[:,slot],2**31-2)+1
    counts=carry.counts.at[:,slot].set(jnp.where(use,count,carry.counts[:,slot]))
    return carry._replace(risk=carry.risk|jnp.where(active,
        value|jnp.where(same,0,NAMESPACE|ORDER)|jnp.where(overflow,OVERFLOW,0)
        |jnp.where((tape.refusal&~public.UNINSTRUMENTED)!=0,UPSTREAM,0),0),
        observed=carry.observed|jnp.where(use,1<<slot,0),counts=counts)


def observe_constructor_device(carry,tape,state,tables,*,active):
    if type(state) is not BattleState:raise ValueError('actual constructor BattleState required')
    b=len(carry.stream_id);bad=(state.turn!=0)|state.done
    for name in ('status','confusion_counter','boosts','volatile','substitute_hp','spikes','wish_counter','encore_counter','charge_counter'):
        x=getattr(state,name);bad |= jnp.any(x!=0,axis=tuple(range(1,x.ndim)))
    bad |= jnp.any(state.trap_source!=-1,axis=(1,2))
    return _observe(carry,tape,tables,Branch.CONSTRUCTOR,jnp.where(bad,CONSTRUCTOR_STATE,0),active=active)


def observe_pre_move_device(carry,tape,state,side,before,tables,*,eligible,active):
    if type(before) is not mechanics.BeforeMove:raise ValueError('literal actual BeforeMove result required')
    b=len(carry.stream_id);side=_vector(side,b);eligible=_vector(eligible,b,jnp.bool_)
    accounting._shape(state.active,(b,2),jnp.int32)
    accounting._shape(state.confusion_counter,(b,2,6),jnp.int32)
    rows=jnp.arange(b);safe_side=jnp.clip(side,0,1);physical=state.active[rows,safe_side]
    address=(side>=0)&(side<2)&(physical>=0)&(physical<6)
    confusion=state.confusion_counter[rows,safe_side,jnp.clip(physical,0,5)]
    bad=eligible&address&(~before.can_act|before.confusion_self_hit|(confusion>0))
    return _observe(carry,tape,tables,Branch.PRE_MOVE,jnp.where(bad,BLOCKED_ACTION,0)
        |jnp.where(eligible&~address,NAMESPACE,0),active=active)


def observe_selected_device(carry,tape,move,tables,*,eligible,active):
    b=len(carry.stream_id);move=_vector(move,b);eligible=_vector(eligible,b,jnp.bool_)
    valid=(move>0)&(move<len(tables.selected_risk));safe=jnp.clip(move,0,len(tables.selected_risk)-1)
    value=jnp.where(eligible,jnp.where(valid,tables.selected_risk[safe],NAMESPACE),0)
    return _observe(carry,tape,tables,Branch.SELECTED_MOVE,value,active=active)


def _captured_direct_matches(tape,outcome,tables,captured):
    """Read-only source receipt match; no risk is cleared retroactively."""
    from . import engine_direct_hit_controls as direct
    b=tape.count.shape[0];no=jnp.zeros(b,bool)
    if captured is None:return no
    if (type(captured) is not CapturedDirectHit or type(captured.context) is not direct.RegisteredContext
            or type(captured.context.data) is not direct.Data or type(captured.write) is not direct.HitWrite):
        raise ValueError('typed SAME-call captured direct hit required')
    context=accounting._context(tape)
    if context is None:return no
    ledger=context.ledger;slot,same=accounting._ticket(ledger,captured.ticket);rows=jnp.arange(b)
    same &= accounting._validate(ledger,captured.context.accounting)
    data=captured.context.data
    if not direct._check(tape,data):return no
    move=_vector(captured.move,b)
    same &= (data.move_count==len(tables.selected_risk))&(move>0)&(move<data.move_count)
    same &= jnp.any(tables.direct_hit_binding!=0)&jnp.all(data.binding==tables.direct_hit_binding)
    same &= jnp.all(captured.context.accounting.inventory_binding==tables.accounting_binding)
    for name in ('engine_binding','source_binding','producer_binding'):
        same &= jnp.all(getattr(data,name)==getattr(tables,name))
        same &= jnp.all(getattr(captured.context.accounting,name)==getattr(context.tables,name))
    same &= jnp.all(data.move_binding==tables.move_binding)&ledger.opened&(ledger.refusal==0)
    same &= (ledger.stage[rows,slot]==2)&(ledger.end[rows,slot]==tape.count)
    same &= accounting._direct_hit_span_matches(tape,captured.ticket,captured.actor,captured.actor_physical,
        captured.target_physical,captured.move,captured.write)
    w=captured.write
    for key in ('power','critical','hit','immune','numerator','denominator','substitute','protected','enduring'):
        same &= getattr(outcome,key)==getattr(w,key)
    start=ledger.start[rows,slot]
    span=(jnp.arange(tape.words.shape[1])[None,:]>=start[:,None])&(
        jnp.arange(tape.words.shape[1])[None,:]<tape.count[:,None])
    same &= jnp.all(~span|((ledger.claimed==slot[:,None]+1)&jnp.all(ledger.words==tape.words,axis=2)
        &(ledger.event_turn==tape.event_turn)&(ledger.event_phase==tape.event_phase)
        &jnp.all(ledger.event_generation==tape.event_generation,axis=2)),axis=1)
    return same


class CapturedSubstitute(NamedTuple):
    context: object
    ticket: object
    side: object
    physical: object
    move: object
    write: object


def _captured_substitute_matches(tape,tables,captured):
    from . import engine_substitute_callbacks as substitute
    b=len(tape.count);no=jnp.zeros(b,bool)
    if captured is None:return no
    if (type(captured) is not CapturedSubstitute or type(captured.context) is not substitute.RegisteredContext
            or type(captured.context.data) is not substitute.Data):
        raise ValueError('typed SAME-call captured Substitute required')
    context=accounting._context(tape)
    if context is None:return no
    ledger=context.ledger;slot,same=accounting._ticket(ledger,captured.ticket);rows=jnp.arange(b)
    same &= accounting._validate(ledger,captured.context.accounting)
    data=captured.context.data
    same &= substitute._check(tape,data)&jnp.any(tables.substitute_binding!=0)
    same &= jnp.all(data.binding==tables.substitute_binding)
    same &= jnp.all(captured.context.accounting.inventory_binding==tables.accounting_binding)
    move=_vector(captured.move,b);same &= (move==data.move_id)&(move>0)&(move<len(tables.substitute_candidates))
    same &= tables.substitute_candidates[jnp.clip(move,0,len(tables.substitute_candidates)-1)]
    for name in ('engine_binding','source_binding','producer_binding'):
        same &= jnp.all(getattr(data,name)==getattr(tables,name))
        same &= jnp.all(getattr(captured.context.accounting,name)==getattr(context.tables,name))
    same &= jnp.all(data.move_binding==tables.move_binding)&ledger.opened&(ledger.refusal==0)
    same &= (ledger.stage[rows,slot]==2)&(ledger.end[rows,slot]==tape.count)
    same &= accounting._substitute_span_matches(tape,captured.ticket,captured.side,captured.physical,
        captured.move,captured.write)
    start=ledger.start[rows,slot];span=(jnp.arange(tape.words.shape[1])[None,:]>=start[:,None])&(
        jnp.arange(tape.words.shape[1])[None,:]<tape.count[:,None])
    same &= jnp.all(~span|((ledger.claimed==slot[:,None]+1)&jnp.all(ledger.words==tape.words,axis=2)
        &(ledger.event_turn==tape.event_turn)&(ledger.event_phase==tape.event_phase)
        &jnp.all(ledger.event_generation==tape.event_generation,axis=2)),axis=1)
    return same


def observe_substitute_device(carry,tape,attempted,tables,*,active,captured=None):
    attempted=_vector(attempted,len(carry.stream_id),jnp.bool_)
    supported=_captured_substitute_matches(tape,tables,captured)
    # Never erase a lifetime bit. This visit adds none only for its OWN fresh
    # exact successful Site11 span; both owner drains remain a separate gate.
    return _observe(carry,tape,tables,Branch.SUBSTITUTE,jnp.where(attempted&~supported,SELF_EFFECT,0),active=active)


def substitute_on_tape(tape,attempted,*,active,captured=None):
    return _on_tape(tape,observe_substitute_device,attempted,active=active,captured=captured)


def observe_outcome_device(carry,tape,outcome,tables,*,eligible,active,captured_direct=None):
    if type(outcome) is not Outcome:raise ValueError('literal typed kernel outcomes required')
    b=len(carry.stream_id);eligible=_vector(eligible,b,jnp.bool_)
    values={name:_vector(x,b,jnp.int32 if name in ('power','numerator','denominator','recoil') else jnp.bool_)
        for name,x in zip(outcome._fields,outcome)}
    o=Outcome(**values)
    unsupported=(o.absorbed|((o.power>0)&(o.critical|~o.hit|o.immune|(o.numerator!=o.denominator)
        |o.substitute|o.protected|o.enduring)))
    direct_mode=jnp.any(tables.direct_hit_binding!=0)
    supported_direct=_captured_direct_matches(tape,o,tables,captured_direct)
    candidate=jnp.zeros(b,bool)
    if type(captured_direct) is CapturedDirectHit:
        move=_vector(captured_direct.move,b)
        candidate=(move>0)&(move<len(tables.direct_candidates))&tables.direct_candidates[
            jnp.clip(move,0,len(tables.direct_candidates)-1)]
    unsupported=jnp.where(direct_mode & ((o.power>0)|candidate),o.absorbed|~supported_direct,unsupported)
    value=jnp.where(unsupported,TARGET_OUTCOME,0)|jnp.where(o.explodes|(o.recoil>0),SELF_HP,0)
    value |= jnp.where((o.power<0)|(o.numerator<0)|(o.denominator<=0)|(o.recoil<0),NAMESPACE,0)
    return _observe(carry,tape,tables,Branch.HIT_OUTCOME,jnp.where(eligible,value,0),active=active)


def observe_branch_device(carry,tape,branch,attempted,tables,*,active):
    """Literal named actual branch guard, NOT accepted/coverage/exhaustive.

    Root must place this at that branch's real cutpoint, including unsupported
    no-write/cap attempts. Calling it with False merely records a visited guard;
    it cannot certify the guard is complete or actually wired correctly.
    """
    if type(branch) is not Branch or int(branch)<6:raise ValueError('named actual unsupported write branch required')
    attempted=_vector(attempted,len(carry.stream_id),jnp.bool_)
    return _observe(carry,tape,tables,branch,jnp.where(attempted,tables.branch_risk[int(branch)],0),active=active)


def observe_end_device(carry,tape,state,tables,*,active):
    return _observe(carry,tape,tables,Branch.FRAME_END,jnp.where(state.done,TERMINAL,0),active=active)


def commit_rows(active,before,after):
    """Explicit GLOBAL8 versus B-row carry; owning Core must actually call it."""
    if type(before) is not Carry or type(after) is not Carry:raise ValueError('typed kernel risk carries required')
    active=_vector(active,len(before.stream_id),jnp.bool_)
    return before._replace(**{name:jnp.where(active.reshape((len(active),)+(1,)*(old.ndim-1)),new,old)
        for name,old,new in zip(before._fields,before,after) if name not in ('binding','accounting_binding')})


def context_on_tape(tape):
    """Absent opt-in is inert; original execution ledger/tables are preserved."""
    execution=getattr(tape,'execution',None)
    context=getattr(execution,'risks',None)
    if context is not None and type(context) is not Context:
        raise ValueError('typed actual kernel risk context required')
    return context


def _on_tape(tape,observe,*args,**kwargs):
    context=context_on_tape(tape)
    if context is None:return tape
    carry=observe(context.carry,tape,*args,context.tables,**kwargs)
    active=_vector(kwargs['active'],len(carry.stream_id),jnp.bool_)
    return tape._replace(execution=tape.execution._replace(risks=context._replace(carry=carry)),
        refusal=tape.refusal|jnp.where(active&(carry.risk!=0),UNKNOWN_KERNEL_RISK,0))


def pre_move_on_tape(tape,state,side,before,*,eligible,active):
    return _on_tape(tape,observe_pre_move_device,state,side,before,eligible=eligible,active=active)


def selected_on_tape(tape,move,*,eligible,active):
    return _on_tape(tape,observe_selected_device,move,eligible=eligible,active=active)


def outcome_on_tape(tape,outcome,*,eligible,active,captured_direct=None):
    return _on_tape(tape,observe_outcome_device,outcome,eligible=eligible,active=active,captured_direct=captured_direct)


def branch_on_tape(tape,branch,attempted,*,active):
    return _on_tape(tape,observe_branch_device,branch,attempted,active=active)


def end_on_tape(tape,state,*,active):
    return _on_tape(tape,observe_end_device,state,active=active)
