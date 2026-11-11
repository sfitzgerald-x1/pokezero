"""Literal original Gen3 Color Change DamagingHit write and public receipt.

RAW43=[43,actor,actorPhysical,target,targetPhysical,move,oldType0,oldType1,newType,ability].
The owning kernel supplies its resolved attack type, not a reconstructed type.
"""
import hashlib,json,subprocess
from pathlib import Path
from typing import NamedTuple
import jax.numpy as j
import numpy as np
from . import engine_public_events as public,engine_lifecycle_callbacks as life

COLOR_CHANGE=43
class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    abilities:object
    candidates:object
    move_types:object
    dynamic_type:object
    installed:object
class Pending(NamedTuple):
    context:object
    ticket:object
    actor:object
    actor_physical:object
    target:object
    physical:object
    move:object
    old:object
    new_type:object
    ability:object
    requested:object
    valid:object

def compile_data(catalog,metadata,rules):
    from . import engine_request_source as source
    from .engine_private_team import engine_metadata_binding
    origin=source._rules_origin(rules)
    if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding) or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):
        raise ValueError('same original Color Change metadata/rules required')
    script=r"""
const fs=require('fs'),path=require('path'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3');
const a=d.abilities.get('colorchange'),lines=[],target={hp:1,isActive:true,side:{active:[{}]},hasType(){return false},setType(type){this.changed=type;return true}};
a.onDamagingHit.call({add(...args){lines.push(args.map(x=>x===target?'TARGET':x))}},1,target,{}, {category:'Special',type:'Water'});
process.stdout.write(JSON.stringify({name:a.name,body:String(a.onDamagingHit),after:a.onAfterMoveSecondary===undefined,lines,changed:target.changed,
source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));"""
    run=subprocess.run([origin.node,'-e',script,origin.root,str(Path(source.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30)
    descriptor=json.loads(run.stdout)
    if json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:
        raise ValueError('original Color Change source changed')
    if not descriptor['after'] or descriptor['name']!='Color Change' or descriptor['changed']!='Water' or descriptor['lines']!=[['-start','TARGET','typechange','Water','[from] ability: Color Change']]:
        raise ValueError('original Gen3 DamagingHit typechange source required')
    t=catalog.engine_table;s=t.secondary
    if s is None:raise ValueError('original installed secondary table required')
    abilities=np.asarray([name=='colorchange' for name in metadata['vocabulary']['abilities']],bool)
    if not np.array_equal(abilities,np.asarray(s.color_change)):raise ValueError('installed Color Change flags differ')
    from . import engine_no_damage_callbacks as no_damage
    candidates=np.asarray(no_damage.compile_data(catalog).candidates).copy()
    dynamic=np.zeros(len(candidates),bool)
    if t.variable_moves is not None:dynamic=np.asarray(t.variable_moves.hidden_power)
    candidates |= dynamic&(np.asarray(catalog.ids)>0)&(np.asarray(catalog.targets)==1)
    payload=dict(schema='pokezero.literal-color-change.v1',descriptor=descriptor,abilities=abilities.tolist(),candidates=candidates.tolist(),types=np.asarray(t.type).tolist(),dynamic=dynamic.tolist(),installed={k:np.asarray(v).tolist()for k,v in zip(s._fields,s)},engine=catalog.engine_binding,source=catalog.source_binding,moves=catalog.move_binding,producer=catalog.binding)
    words=lambda v:j.asarray(v,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(abilities),j.asarray(candidates),t.type,j.asarray(dynamic),s)

def shape(d):
    if type(d)is not Data:raise ValueError('original typed Color Change data required')
    for x in d[:5]:life._typed(x,(8,),j.uint32)
    life._typed(d.abilities,d.installed.color_change.shape,j.bool_)
    life._typed(d.candidates,d.move_types.shape,j.bool_);life._typed(d.move_types,d.candidates.shape,j.int32);life._typed(d.dynamic_type,d.candidates.shape,j.bool_)

def _bound(tape,context):
    from . import engine_direct_hit_controls as direct
    if context is None or type(context)is not direct.RegisteredContext or context.data.color_change is None:return j.zeros(len(tape.count),bool)
    d=context.data.color_change;shape(d);good=direct._registered_bound(tape,context)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);good &= j.all(getattr(d,name)==getattr(tape,name))
    return good

def begin(tape,state,side,move,typ,context,table,*,eligible):
    from . import engine_execution_completeness as ex
    if context is None or context.data.color_change is None:
        return tape._replace(refusal=tape.refusal|j.where(eligible,256,0)),None
    b=len(tape.count);rows=j.arange(b);d=context.data.color_change;shape(d)
    for x in (side,move,typ):life._typed(x,(b,),j.int32)
    life._typed(eligible,(b,),j.bool_)
    good=_bound(tape,context)
    if type(table)is not type(d.installed):raise ValueError('original Color Change setter table required')
    for a,z in zip(table,d.installed,strict=True):
        if a.shape!=z.shape or a.dtype!=z.dtype:raise ValueError('Color Change installed shape differs')
        good &= j.all(a==z)
    target=1-side;ap=state.active[rows,side];physical=state.active[rows,target]
    old=state.types[rows,target,physical];abi=state.ability[rows,target,physical]
    safe=j.clip(move,0,len(d.candidates)-1)
    good &= (move>0)&(move<len(d.candidates))&d.candidates[safe]
    good &= (typ>0)&(typ<18)&(typ==j.where(d.dynamic_type[safe],state.hidden_power_type[rows,side,ap],d.move_types[safe]))
    good &= (abi>0)&(abi<len(d.abilities))&d.abilities[j.clip(abi,0,len(d.abilities)-1)]
    good &= (old[:,0]>0)&j.all((old>=0)&(old<18),axis=1)&~j.any(old==typ[:,None],axis=1)
    good &= (state.hp[rows,target,physical]>0)&j.all(state.base_types[rows,target,physical]>=0,axis=1)
    good &= (ap==tape.cursor.physical_active[rows,side])&(physical==tape.cursor.physical_active[rows,target])&(tape.cursor.phase==0)
    tape,ticket=ex.open_attempt(tape,ex.Site.COLOR_CHANGE,target,active=eligible)
    return tape,Pending(context,ticket,side,ap,target,physical,move,old,typ,abi,eligible,good)

def _receipt(tape,index,*,stage=2,data=None):
    from . import engine_execution_completeness as ex
    b=len(tape.count);rows=j.arange(b);at=j.broadcast_to(j.asarray(index,j.int32),(b,));safe=j.clip(at,0,tape.words.shape[1]-1);raw=tape.words[rows,safe];c=ex._context(tape)
    if c is None or c.risks is None:return raw,j.zeros(b,bool)
    ledger=c.ledger;claim=ledger.count if stage==1 else ledger.claimed[rows,safe];slot=j.clip(claim-1,0,ledger.site.shape[1]-1)
    good=ex._validate(ledger,c.tables)&ex._tape(ledger,tape,c.tables)&ledger.opened
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32)
        if data is not None:good &= j.all(getattr(data,name)==getattr(tape,name))
    good &= (at>0)&(at<tape.count)&(claim>0)&(claim<=ledger.count)&(raw[:,0]==COLOR_CHANGE)
    good &= (raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,3]==1-raw[:,1])&(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,4]>=0)&(raw[:,4]<6)
    good &= (raw[:,5]>0)&(raw[:,6]>0)&j.all((raw[:,6:9]>=0)&(raw[:,6:9]<18),axis=1)&(raw[:,8]>0)&~j.any(raw[:,6:8]==raw[:,8,None],axis=1)&(raw[:,9]>0)
    good &= (ledger.site[rows,slot]==int(ex.Site.COLOR_CHANGE))&(ledger.side[rows,slot]==raw[:,3])&(ledger.stage[rows,slot]==stage)&(ledger.start[rows,slot]==at)
    good &= j.where(stage==1,tape.count==at+1,ledger.end[rows,slot]==at+1)
    # The latest actual MOVE must own the same actor, target, move and clocks.
    positions=j.arange(tape.words.shape[1])[None,:];prior=(positions<at[:,None])&(tape.words[:,:,0]==public.MOVE)
    prefix=j.max(j.where(prior,positions,-1),axis=1);p=j.clip(prefix,0,tape.words.shape[1]-1);decl=tape.words[rows,p]
    good &= (prefix>=0)&j.all(decl[:,1:5]==raw[:,1:5],axis=1)&(decl[:,5]==raw[:,5])&j.all(decl[:,6:]==0,axis=1)
    # Positive damage is the original SAME-call Site10 HP-write capability,
    # never inferred from an alive afterstate or a caller's success flag.
    entries=j.arange(ledger.site.shape[1])[None,:]<ledger.count[:,None]
    hits=entries&(ledger.site==int(ex.Site.DIRECT_HIT))&(ledger.stage==2)&(ledger.side==raw[:,3,None])
    hits &= (ledger.start==prefix[:,None]+1)&(ledger.end<=at[:,None])&(ledger.end>ledger.start)&(ledger.no_output_kind==0)
    hit_slot=j.argmax(hits,axis=1);end=ledger.end[rows,hit_slot]-1;hp_at=j.clip(end,0,tape.words.shape[1]-1);hp=tape.words[rows,hp_at]
    good &= j.sum(hits,axis=1)==1
    good &= (hp[:,0]==public.MAIN_HP)&(hp[:,3]==raw[:,3])&(hp[:,4]==raw[:,4])&(hp[:,5]>0)&(hp[:,6]>=hp[:,5])
    good &= (ledger.claimed[rows,hp_at]==hit_slot+1)&j.all(ledger.words[rows,hp_at]==hp,axis=1)
    good &= (ledger.turn[rows,hit_slot]==ledger.turn[rows,slot])&(ledger.phase[rows,hit_slot]==ledger.phase[rows,slot])&j.all(ledger.generation[rows,hit_slot]==ledger.generation[rows,slot],axis=1)
    for pos in (p,safe):good &= (tape.event_turn[rows,pos]==ledger.turn[rows,slot])&(tape.event_phase[rows,pos]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,pos]==ledger.generation[rows,slot],axis=1)
    if stage==2:good &= j.all(ledger.words[rows,safe]==raw,axis=1)&(ledger.event_turn[rows,safe]==tape.event_turn[rows,safe])&(ledger.event_phase[rows,safe]==tape.event_phase[rows,safe])&j.all(ledger.event_generation[rows,safe]==tape.event_generation[rows,safe],axis=1)
    r=c.risks.tables
    life._typed(r.color_change_binding,(8,),j.uint32)
    life._typed(r.color_change_abilities,(len(r.color_change_abilities),),j.bool_)
    for value,dtype in ((r.color_change_candidates,j.bool_),(r.color_change_types,j.int32),(r.color_change_dynamic,j.bool_)):life._typed(value,r.selected_risk.shape,dtype)
    good &= j.any(r.color_change_binding!=0)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(r,name),(8,),j.uint32);good &= j.all(getattr(r,name)==getattr(tape,name))
    mv=j.clip(raw[:,5],0,len(r.color_change_candidates)-1);abi=j.clip(raw[:,9],0,len(r.color_change_abilities)-1)
    good &= (raw[:,5]<len(r.color_change_candidates))&r.color_change_candidates[mv]&(raw[:,9]<len(r.color_change_abilities))&r.color_change_abilities[abi]
    good &= r.color_change_dynamic[mv]|(r.color_change_types[mv]==raw[:,8])
    if data is not None:
        good &= j.all(r.color_change_binding==data.binding)
        shape(data);mv=j.clip(raw[:,5],0,len(data.candidates)-1);abi=j.clip(raw[:,9],0,len(data.abilities)-1)
        good &= (raw[:,5]<len(data.candidates))&data.candidates[mv]&(raw[:,9]<len(data.abilities))&data.abilities[abi]
        good &= data.dynamic_type[mv]|(data.move_types[mv]==raw[:,8])
    return raw,good

def owned(tape,index,data=None):return _receipt(tape,index,data=data)[1]

def finish(tape,p,state):
    if p is None:return tape
    from . import engine_execution_completeness as ex
    rows=j.arange(len(tape.count));new=state.types[rows,p.target,p.physical]
    good=p.valid&_bound(tape,p.context)&(new[:,0]==p.new_type)&(new[:,1]==0)&(state.active[rows,p.target]==p.physical)&(state.ability[rows,p.target,p.physical]==p.ability)&(state.hp[rows,p.target,p.physical]>0)
    tape=public._append(tape,COLOR_CHANGE,p.actor,p.actor_physical,p.target,p.physical,(p.move,p.old[:,0],p.old[:,1],p.new_type,p.ability),p.requested&good)
    raw,receipt=_receipt(tape,tape.count-1,stage=1,data=p.context.data.color_change);c=ex._context(tape);slot,same=ex._ticket(c.ledger,p.ticket)
    good &= receipt&same&(c.ledger.start[rows,slot]==tape.count-1)
    expected=j.stack((j.full_like(p.actor,COLOR_CHANGE),p.actor,p.actor_physical,p.target,p.physical,p.move,p.old[:,0],p.old[:,1],p.new_type,p.ability),axis=1)
    good &= j.all(raw==expected,axis=1)
    ledger=ex.emitted(c.ledger,p.ticket,tape,active=p.requested&good,tables=c.tables);ledger=ex.unhandled(ledger,p.ticket,active=p.requested&~good,tables=c.tables)
    return ex._put_context(tape._replace(refusal=tape.refusal|j.where(p.requested&~good,256,0)),ledger,c)

def expected_owned_types(engine,before_engine,public_state,*,owner,tape,data,categories,type_tables,healthy):
    """Trusted owning-driver join, before requests and before the new FULL fold.

    Start from sealed constructor types. Only the exact previous public active
    encounter or this frame's literal claimed Color Change can override it.
    Bench members always retain their constructor types. These inputs are the
    original owning Session's carries, not a public attestation API.
    """
    from . import public_types
    b=len(tape.count);rows=j.arange(b);shape(data)
    life._typed(categories,(18,),j.int32);life._typed(healthy,(b,),j.bool_)
    expected=engine.base_types[rows,owner]
    old=public_state.context.types;mons=public_state.mons
    public_types._shape(old,type_tables,mons)
    current,_,coherent=public_types._coherent(old,type_tables,mons)
    category=old.categories[rows,owner]
    matches=category[:,:,None]==categories[None,None,:]
    counts=j.sum(matches,axis=2)
    mapped=j.argmax(matches,axis=2).astype(j.int32)
    known=j.all(counts==1,axis=1)&(mapped[:,0]>0)&(mapped[:,1]==0)
    use=old.overridden[rows,owner]&(engine.active[rows,owner]==before_engine.active[rows,owner])
    prior_ok=healthy&coherent&(old.encounter[rows,owner]==current[rows,owner])&known
    prior=j.where(prior_ok[:,None],mapped,-j.ones_like(mapped))
    active=engine.active[rows,owner]
    expected=expected.at[rows,active].set(j.where(use[:,None],prior,expected[rows,active]))
    def observe(index,value):
        raw,good=_receipt(tape,index,data=data)
        applies=good&(index<tape.count)&(raw[:,3]==owner)&(raw[:,4]==active)
        same_old=j.all(value[rows,active]==raw[:,6:8],axis=1)
        pair=j.stack((raw[:,8],j.zeros(b,j.int32)),axis=1)
        pair=j.where(same_old[:,None],pair,-j.ones_like(pair))
        return value.at[rows,active].set(j.where(applies[:,None],pair,value[rows,active]))
    import jax
    return jax.lax.fori_loop(0,tape.words.shape[1],observe,expected)
