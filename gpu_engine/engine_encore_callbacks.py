"""Original Encore apply/residual transactions, RAW48/Site34.

START/END are literal public facts. TICK is source-proven internal duration
transport with no public word. FAIL_NO_LAST is only the literal unchanged
no-prior-move failure. Device carries are trusted owning-callsite inputs, not
copy-resistant origin certificates. Other failed/called/ambiguous writes remain
UNKNOWN; no duration is disclosed in public packets.
"""
import hashlib,json,subprocess
from pathlib import Path
from typing import NamedTuple
import jax
import jax.numpy as j
import numpy as np
from . import engine_public_events as public,engine_lifecycle_callbacks as life

ENCORE=48
START,END,TICK=range(3)
FAIL_NO_LAST=3
RUN_MOVE_EFFECTS_SHA256='bb75053d0bc1c666f41ea43143e8440ec1d415194e979d79eb007ae7ed3ea2b5'
HASHES={'durationCallback':'5fa95c56ee1187568ff774129ba7816873d4d6a4806130268725463caea45786','onStart':'65cca25468443dea05dd74c855318f3668d29f5a791838ef10d845536937eeca','onEnd':'b44a4536112a28dc5d572d3f77186bfd6290da2bafadd437339594765fe0928b','onResidual':'0632f332d720858a486eccf54c96f01311485a4d84db53bb4bde9e7bc36011e8','onDisableMove':'85258aafd68ffe84c3ffe47548d3b8335d6e8989fe4d143ecee987416d21deb8','onOverrideAction':'b843dacd8a49e1eeb2a0642e0ce357047ff21de3bc9dc2af86dfcd9d48de9f7a'}
class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    move_id:object
    installed:object
class NoLastBefore(NamedTuple):
    last_move:object
    kind:object
    index:object
    moves:object
    pp:object
    hp:object
    maximum:object
    status:object
    status_counter:object
class Pending(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    old_move:object
    old_counter:object
    recorded_slot:object
    recorded_move:object
    pp:object
    hp:object
    expected_counter:object
    residual:object
    secondary:object
    requested:object
    valid:object
    failure:object=None
    before:object=None
class Captured(NamedTuple):
    pending:object
    raw:object
class Request(NamedTuple):
    engine_binding:object
    rules_binding:object
    stream_id:object
    owner:object
    physical:object
    move:object
    slot:object
    counter:object
    turn:object
    generation:object

def compile_data(catalog,metadata,rules):
    from . import engine_request_source as request
    from .engine_private_team import engine_metadata_binding
    origin=request._rules_origin(rules)
    if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding)or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):raise ValueError('same original Encore metadata/rules required')
    script=r"""
const fs=require('fs'),path=require('path'),crypto=require('crypto'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3'),m=d.moves.get('encore'),c=m.condition;
const runMoveEffects=require(path.join(root,'dist/sim/battle-actions.js')).BattleActions.prototype.runMoveEffects;
const hashes={};for(const k of ['durationCallback','onStart','onEnd','onResidual','onDisableMove','onOverrideAction'])hashes[k]=crypto.createHash('sha256').update(c[k].toString()).digest('hex');
const rows=meta.vocabulary.moves.map(id=>{const x=d.moves.get(id);return {id:x.id,exists:x.exists,gen:x.gen,failencore:!!x.flags.failencore}});
process.stdout.write(JSON.stringify({rows,hashes,runMoveEffects:crypto.createHash('sha256').update(runMoveEffects.toString()).digest('hex'),order:c.onResidualOrder,suborder:c.onResidualSubOrder,category:m.category,target:m.target,power:m.basePower,accuracy:m.accuracy,volatile:m.volatileStatus,source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));"""
    descriptor=json.loads(subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30).stdout)
    if descriptor['runMoveEffects']!=RUN_MOVE_EFFECTS_SHA256:raise ValueError('original failed Encore writer required')
    if descriptor['hashes']!=HASHES or descriptor['order']!=10 or descriptor['suborder']!=14 or (descriptor['category'],descriptor['target'],descriptor['power'],descriptor['accuracy'],descriptor['volatile'])!=('Status','normal',0,100,'encore') or json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original Gen3 Encore source changed')
    table=catalog.engine_table.encore
    if table is None:return None
    names=metadata['vocabulary']['moves'];mask=np.array([n=='encore'for n in names]);allowed=np.array([bool(i and r['exists']and r['gen']<=3 and (r['id']==names[i]or(r['id']=='hiddenpower'and names[i].startswith('hiddenpower')))and names[i]not in('struggle','encore')and not r['failencore'])for i,r in enumerate(descriptor['rows'])])
    if not mask.any():return None
    if not np.array_equal(table.move,mask)or not np.array_equal(table.allowed_last,allowed):raise ValueError('original installed Encore table required')
    ids=np.flatnonzero(mask);move=int(ids[0])if len(ids)==1 else 0
    if move and (int(catalog.ids[move])<=0 or int(catalog.targets[move])!=1 or int(catalog.engine_table.power[move])!=0 or int(catalog.engine_table.accuracy[move])!=100 or bool(catalog.engine_table.always_hit[move])):raise ValueError('original Encore move row required')
    payload=dict(schema='pokezero.literal-encore-transaction.v2',descriptor=descriptor,engine=catalog.engine_binding,source=catalog.source_binding,moves=catalog.move_binding,producer=catalog.binding,installed={k:np.asarray(v).tolist()for k,v in zip(table._fields,table)})
    words=lambda x:j.asarray(x,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(move,j.int32),table)

def shape(d):
    if type(d)is not Data:raise ValueError('typed original Encore data required')
    for x in d[:5]:life._typed(x,(8,),j.uint32)
    life._typed(d.move_id,(),j.int32);life._typed(d.installed.move,d.installed.allowed_last.shape,j.bool_);life._typed(d.installed.allowed_last,d.installed.move.shape,j.bool_)

def _bound(tape,context):
    from . import engine_direct_hit_controls as direct
    d=context.data.encore;shape(d);good=direct._registered_bound(tape,context)
    if tape.execution is None or tape.execution.risks is None:return j.zeros(len(tape.count),bool)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);good &= j.all(getattr(d,name)==getattr(tape,name))
    t=tape.execution.risks.tables;life._typed(t.encore_binding,(8,),j.uint32);life._typed(t.encore_move,(),j.int32)
    return good&j.any(d.binding!=0)&j.all(t.encore_binding==d.binding)&(t.encore_move==d.move_id)&life._valid(tape.cursor)&tape.cursor.initialized

def begin(tape,state,side,move,context,table,*,eligible,residual=False,secondary=None,roll=None,queued=None):
    from . import engine_execution_completeness as ex,action_history
    if context is None or context.data.encore is None:return tape,None
    b=len(tape.count);rows=j.arange(b);d=context.data.encore;good=_bound(tape,context)
    for x in(side,move):life._typed(x,(b,),j.int32)
    life._typed(eligible,(b,),j.bool_)
    if secondary is None:secondary=j.zeros(b,bool)
    life._typed(secondary,(b,),j.bool_)
    if type(residual)is not bool:raise ValueError('static original Encore callsite required')
    if type(table)is not type(d.installed):raise ValueError('original installed Encore table required')
    for x,y in zip(table,d.installed,strict=True):
        if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('Encore table geometry changed')
        good &= j.all(x==y)
    seat=j.clip(side,0,1);physical=state.active[rows,seat];at=(rows,seat,physical);slot,valid=action_history.move_slot(state,seat);last=state.moves[rows,seat,physical,slot];pp=state.pp[rows,seat,physical,slot]
    old=state.encore_move[at];counter=state.encore_counter[at]
    if residual:expected_counter=j.where((counter<=1)|(pp<=0),0,counter-1)
    else:
        if roll is None or queued is None:raise ValueError('literal Encore roll/queue required')
        life._typed(roll,(b,),j.int32);life._typed(queued,(b,),j.bool_)
        good &= (roll>=0)&(roll<4)
        expected_counter=3+roll+(~queued).astype(j.int32)
    requested=eligible&((counter>0)if residual else(move==d.move_id)&(d.move_id>0))
    common=good
    no_last=j.zeros(b,bool)
    before=NoLastBefore(state.last_move[at],state.last_action_kind[rows,seat],state.last_action_index[rows,seat],state.moves[at],state.pp[at],state.hp[at],state.max_hp[at],state.status[at],state.status_counter[at])
    if not residual:
        no_last=(before.last_move==0)&(((before.kind==action_history.NONE)&(before.index==-1))|((before.kind==action_history.SWITCH)&(before.index==physical)))
        no_last &= common&requested&secondary&(counter==0)&(old==0)&(state.hp[at]>0)&(state.hp[at]<=state.max_hp[at])&(side>=0)&(side<2)&(physical>=0)&(physical<6)&(physical==tape.cursor.physical_active[rows,seat])&(tape.cursor.phase==0)
    good &= (side>=0)&(side<2)&(physical>=0)&(physical<6)&(physical==tape.cursor.physical_active[rows,seat])&(state.hp[at]>0)&valid&(last>0)&(last<len(table.allowed_last))
    good &= table.allowed_last[j.clip(last,0,len(table.allowed_last)-1)]&(j.sum(state.moves[rows,seat,physical]==last[:,None],axis=1)==1)
    good &= ((counter>0)&(counter<=7)&(old==last))if residual else((counter==0)&(old==0)&(pp>0)&(tape.cursor.phase==0))
    good |= no_last
    expected_counter=j.where(no_last,0,expected_counter)
    tape,ticket=ex.open_attempt(tape,ex.Site.ENCORE,side,active=requested)
    return tape,Pending(context,ticket,side,physical,old,counter,slot,last,pp,state.hp[at],expected_counter,j.full(b,residual),secondary,requested,good,no_last,before)

def valid_raw(raw,move_count):
    start=raw[:,3]==START;end=raw[:,3]==END;tick=raw[:,3]==TICK
    failure=(raw[:,3]==FAIL_NO_LAST)&j.all(raw[:,4:9]==0,axis=1)&((raw[:,9]==0)|(raw[:,9]==1))
    return ((raw[:,0]==ENCORE)&(raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,4]>0)&(raw[:,4]<move_count)&(raw[:,9]>=0)&(raw[:,9]<8)&((raw[:,3]==START)|(raw[:,9]<4))
        &((start&(raw[:,5]==0)&(raw[:,6]==raw[:,4])&(raw[:,7]==0)&(raw[:,8]>=3)&(raw[:,8]<=7))
          |(end&(raw[:,5]==raw[:,4])&(raw[:,6]==0)&(raw[:,7]>=1)&(raw[:,7]<=7)&(raw[:,8]==0))
          |(tick&(raw[:,5]==raw[:,4])&(raw[:,6]==raw[:,4])&(raw[:,7]>1)&(raw[:,7]<=7)&(raw[:,8]==raw[:,7]-1)))) | ((raw[:,0]==ENCORE)&(raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,2]>=0)&(raw[:,2]<6)&failure)

def owned(tape,index,*,stage=2):
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=j.arange(b);idx=j.broadcast_to(j.asarray(index,j.int32),(b,));at=j.clip(idx,0,tape.words.shape[1]-1);raw=tape.words[rows,at]
    if c is None or c.risks is None:return j.zeros(b,bool)
    t=c.risks.tables;ledger=c.ledger;life._typed(t.encore_binding,(8,),j.uint32);life._typed(t.encore_move,(),j.int32)
    good=ex._validate(ledger,c.tables)&ex._tape(ledger,tape,c.tables)&ledger.opened&(ledger.frame>0)&j.any(t.encore_binding!=0)&(t.encore_move>0)
    for name in('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);life._typed(getattr(t,name),(8,),j.uint32);good &= j.all(getattr(tape,name)==getattr(t,name))
    claim=ledger.count if stage==1 else ledger.claimed[rows,at];slot=j.clip(claim-1,0,ledger.site.shape[1]-1)
    good &= valid_raw(raw,len(t.selected_risk))&(idx>=0)&(idx<tape.count)&(claim>0)&(claim<=ledger.count)&j.all(tape.switch_details[rows,at]==0,axis=1)
    good &= (ledger.site[rows,slot]==int(ex.Site.ENCORE))&(ledger.side[rows,slot]==raw[:,1])&(ledger.stage[rows,slot]==stage)&(ledger.start[rows,slot]==idx)&(ledger.no_output_kind[rows,slot]==0)
    good &= j.where(stage==1,tape.count==idx+1,ledger.end[rows,slot]==idx+1)
    good &= (tape.event_turn[rows,at]==ledger.turn[rows,slot])&(tape.event_phase[rows,at]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,at]==ledger.generation[rows,slot],axis=1)
    prefix=j.clip(idx-1,0,tape.words.shape[1]-1);decl=tape.words[rows,prefix]
    start=(raw[:,3]==START)|(raw[:,3]==FAIL_NO_LAST)
    good &= ~start|((idx>0)&(decl[:,0]==public.MOVE)&(decl[:,1]==1-raw[:,1])&(decl[:,3]==raw[:,1])&(decl[:,4]==raw[:,2])&(decl[:,5]==t.encore_move)&j.all(decl[:,6:]==0,axis=1)&(tape.event_turn[rows,prefix]==ledger.turn[rows,slot])&(tape.event_phase[rows,prefix]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,prefix]==ledger.generation[rows,slot],axis=1))
    if stage==1:good &= (~start|(decl[:,2]==tape.cursor.physical_active[rows,j.clip(1-raw[:,1],0,1)]))&(raw[:,2]==tape.cursor.physical_active[rows,j.clip(raw[:,1],0,1)])&(ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)&j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
    else:good &= (ledger.claimed[rows,at]==claim)&j.all(ledger.words[rows,at]==raw,axis=1)&(ledger.event_turn[rows,at]==tape.event_turn[rows,at])&(ledger.event_phase[rows,at]==tape.event_phase[rows,at])&j.all(ledger.event_generation[rows,at]==tape.event_generation[rows,at],axis=1)
    fail=raw[:,3]==FAIL_NO_LAST
    facts=ledger.no_output_facts[rows,slot]
    expected=j.stack((j.zeros(b,j.int32),raw[:,9],j.where(raw[:,9]==0,-1,raw[:,2]),facts[:,3],facts[:,4],facts[:,5]),axis=1)
    failure_good=j.all(facts==expected,axis=1)&(facts[:,3]>0)&(facts[:,4]>=0)&(facts[:,4]<=6)&(facts[:,5]>=0)&(facts[:,5]<=15)&j.all(ledger.no_output_binding[rows,slot]==t.encore_binding[None,:],axis=1)
    return good&(~fail|failure_good)

def _failure(p):
    if p.failure is None:return j.zeros_like(p.requested)
    life._typed(p.failure,p.requested.shape,j.bool_)
    return p.failure

def _failure_raw(p):
    z=j.zeros_like(p.side)
    return j.stack((z+ENCORE,p.side,p.physical,z+FAIL_NO_LAST,z,z,z,z,z,p.before.kind),axis=1)

def _failure_facts(p):
    a=p.before
    return j.stack((a.last_move,a.kind,a.index,a.hp,a.status,a.status_counter),axis=1)

def finish(tape,p,state):
    if p is None:return tape,None
    from . import engine_execution_completeness as ex
    b=len(tape.count);rows=j.arange(b);at=(rows,j.clip(p.side,0,1),p.physical);new=state.encore_move[at];counter=state.encore_counter[at]
    code=j.where(p.residual,j.where(counter==0,END,TICK),START).astype(j.int32)
    raw=j.stack((j.full(b,ENCORE,j.int32),p.side,p.physical,code,p.recorded_move,p.old_move,new,p.old_counter,counter,p.recorded_slot+4*p.secondary.astype(j.int32)),axis=1)
    failure=_failure(p)
    if p.before is not None:raw=j.where(failure[:,None],_failure_raw(p),raw)
    valid=p.valid&_bound(tape,p.context)&valid_raw(raw,len(p.context.data.encore.installed.move))&(state.active[rows,j.clip(p.side,0,1)]==p.physical)
    valid &= (counter==p.expected_counter)&(state.hp[at]==p.hp)
    positive=(state.moves[rows,j.clip(p.side,0,1),p.physical,p.recorded_slot]==p.recorded_move)&(state.pp[rows,j.clip(p.side,0,1),p.physical,p.recorded_slot]==p.pp)&(state.last_action_kind[rows,j.clip(p.side,0,1)]==2)&(state.last_action_index[rows,j.clip(p.side,0,1)]==p.recorded_slot)
    same=j.zeros(b,bool)
    if p.before is not None:
        a=p.before
        for x in(a.last_move,a.kind,a.index,a.hp,a.maximum,a.status,a.status_counter):life._typed(x,(b,),j.int32)
        life._typed(a.moves,(b,4),j.int32);life._typed(a.pp,(b,4),j.int32)
        same=(state.last_move[at]==a.last_move)&(state.last_action_kind[rows,j.clip(p.side,0,1)]==a.kind)&(state.last_action_index[rows,j.clip(p.side,0,1)]==a.index)&j.all(state.moves[at]==a.moves,axis=1)&j.all(state.pp[at]==a.pp,axis=1)&(state.hp[at]==a.hp)&(state.max_hp[at]==a.maximum)&(state.status[at]==a.status)&(state.status_counter[at]==a.status_counter)
    valid &= j.where(failure,same,positive)
    room=(tape.count>=0)&(tape.count<tape.words.shape[1]);address=j.clip(tape.count,0,tape.words.shape[1]-1);valid &= ~room|j.all(tape.switch_details[rows,address]==0,axis=1)
    tape=public._append(tape,ENCORE,p.side,p.physical,raw[:,3],raw[:,4],tuple(raw[:,i]for i in range(5,10)),p.requested&valid&room)
    c=ex._context(tape);slot,same=ex._ticket(c.ledger,p.ticket)
    if p.before is not None:
        use=p.requested&valid&room&failure
        ledger=c.ledger._replace(no_output_facts=c.ledger.no_output_facts.at[rows,slot].set(j.where(use[:,None],_failure_facts(p),c.ledger.no_output_facts[rows,slot])),no_output_binding=c.ledger.no_output_binding.at[rows,slot].set(j.where(use[:,None],p.context.data.encore.binding[None,:],c.ledger.no_output_binding[rows,slot])))
        tape=ex._put_context(tape,ledger,c);c=ex._context(tape)
    good=owned(tape,j.maximum(tape.count-1,0),stage=1)&valid&room&same&(c.ledger.start[rows,slot]==tape.count-1)
    ledger=ex.unhandled(c.ledger,p.ticket,active=p.requested&~good,tables=c.tables);ledger=ex.emitted(ledger,p.ticket,tape,active=p.requested&good,tables=c.tables);tape=ex._put_context(tape,ledger,c)
    tape=tape._replace(refusal=tape.refusal|j.where(p.requested&~good,256,0)|j.where(p.requested&valid&~room,public.OVERFLOW,0))
    return tape,Captured(p,raw)

def captured_matches(tape,c):
    if c is None:return j.zeros(len(tape.count),bool)
    if type(c)is not Captured or type(c.pending)is not Pending:raise ValueError('literal same-call Encore capture required')
    from . import engine_execution_completeness as ex
    p=c.pending;rows=j.arange(len(tape.count));ledger=tape.execution.ledger;slot,same=ex._ticket(ledger,p.ticket);idx=ledger.start[rows,slot];raw=tape.words[rows,j.clip(idx,0,tape.words.shape[1]-1)]
    failure=_failure(p);joined=j.zeros(len(p.side),bool)
    if p.before is not None:
        a=p.before
        from . import action_history
        joined=(a.last_move==0)&(((a.kind==action_history.NONE)&(a.index==-1))|((a.kind==action_history.SWITCH)&(a.index==p.physical)))&(p.old_move==0)&(p.old_counter==0)&(p.expected_counter==0)&~p.residual&p.secondary
        selected=j.clip(a.index,0,3)
        joined &= (p.hp==a.hp)&(p.recorded_slot==selected)&(p.recorded_move==a.moves[rows,selected])&(p.pp==a.pp[rows,selected])
        joined &= j.all(raw==_failure_raw(p),axis=1)&j.all(ledger.no_output_facts[rows,slot]==_failure_facts(p),axis=1)
    return ((raw[:,3]==FAIL_NO_LAST)==failure)&(~failure|joined)&owned(tape,idx)&same&p.requested&p.valid&_bound(tape,p.context)&j.all(raw==c.raw,axis=1)&(ledger.end[rows,slot]==tape.count)&(ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)&j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)&(p.physical==tape.cursor.physical_active[rows,j.clip(p.side,0,1)])

def expected_request(engine,before,public_state,*,owner,tape,data,active_tables,rules,healthy):
    """Owning-only public history + SAME literal transaction -> forced-slot facts.

    No private peer facts or absence-of-output proof. A positive current lock
    requires either the prior original Encore public encounter or a fresh
    claimed START, then the exact latest source transaction/current counter.
    Missing history/duplicate moves/PP exhaustion refuse, rather than guess.
    """
    from . import public_active,action_history
    shape(data);b=len(tape.count);rows=j.arange(b);life._typed(healthy,(b,),j.bool_)
    old=public_state.context.active;public_active._shape(old,active_tables,public_state.mons)
    current,coherent=public_active._coherent(old,active_tables,public_state.mons)
    physical=engine.active[:,owner];safe=j.clip(physical,0,5);at=(rows,j.full(b,owner,j.int32),safe)
    slot,recorded=action_history.move_slot(engine,j.full(b,owner,j.int32));move=engine.encore_move[at];counter=engine.encore_counter[at]
    present=old.volatiles[:,owner,public_active.NAMES.index('encore')]
    established=healthy&coherent&(old.encounter[:,owner]==current[:,owner])&present&(physical==before.active[:,owner])
    # Track the literal source stream, including START followed by residual
    # ticks in this frame. END closes it; an unproved record poisons the join.
    expected_move=j.where(established,before.encore_move[rows,owner,j.clip(before.active[:,owner],0,5)],0)
    expected_counter=j.where(established,before.encore_counter[rows,owner,j.clip(before.active[:,owner],0,5)],0)
    def observe(index,value):
        m,c,seen=value;raw=tape.words[:,index];applies=(index<tape.count)&(raw[:,0]==ENCORE)&(raw[:,1]==owner)&(raw[:,2]==physical)
        good=owned(tape,index)&j.all(tape.engine_binding==data.engine_binding)&j.all(tape.source_binding==data.source_binding)&j.all(tape.move_binding==data.move_binding)&j.all(tape.producer_binding==data.producer_binding)
        good &= (raw[:,5]==m)&(raw[:,7]==c)
        applies &= ~((raw[:,3]==FAIL_NO_LAST)&good)
        return j.where(applies,j.where(good,raw[:,6],-1),m),j.where(applies,j.where(good,raw[:,8],-1),c),seen|applies
    expected_move,expected_counter,seen=jax.lax.fori_loop(0,tape.words.shape[1],observe,(expected_move,expected_counter,j.zeros(b,bool)))
    # Current-frame residual capture is mandatory for a live retained lock.
    moves=engine.moves[rows,owner,safe];pp=engine.pp[rows,owner,safe,slot]
    good=healthy&seen&recorded&(move>0)&(move<len(data.installed.allowed_last))&data.installed.allowed_last[j.clip(move,0,len(data.installed.allowed_last)-1)]
    good &= (counter>=1)&(counter<=7)&(move==expected_move)&(counter==expected_counter)&(moves[rows,slot]==move)&(pp>0)&(j.sum(moves==move[:,None],axis=1)==1)&(engine.hp[at]>0)&(physical>=0)&(physical<6)
    selected=j.where((move==0)&(counter==0),-1,j.where(good,slot,-2)).astype(j.int32)
    return Request(data.engine_binding,rules.binding,tape.stream_id,j.full(b,owner,j.int32),physical,move,selected,counter,engine.turn,tape.cursor.generation)

def request_slot(value,state,ledger,rules,arrival):
    """Validate typed exposure against the actual original server request."""
    b=len(ledger.stream_id);rows=j.arange(b)
    if value is None:return j.full(b,-1,j.int32),j.zeros(b,bool)
    if type(value)is not Request:raise ValueError('typed original Encore request exposure required')
    for x in value[:2]:life._typed(x,(8,),j.uint32)
    for x in value[2:-1]:life._typed(x,(b,),j.int32)
    life._typed(value.generation,(b,2),j.int32)
    owner=j.clip(ledger.owner_side,0,1);active=state.active[rows,owner];safe=j.clip(active,0,5);at=(rows,owner,safe);slot=j.clip(value.slot,0,3)
    good=j.all(value.engine_binding==rules.engine_binding)&j.all(value.rules_binding==rules.binding)&j.all(ledger.engine_binding==rules.engine_binding,axis=1)&(value.stream_id==ledger.stream_id)&(value.owner==ledger.owner_side)&(value.physical==active)
    good &= (value.turn==state.turn)&j.all(value.generation==arrival.generation,axis=1)&(value.generation[rows,owner]==ledger.generation)
    good &= (value.slot>=0)&(value.slot<4)&(value.move==state.encore_move[at])&(value.counter==state.encore_counter[at])&(value.counter>0)&(state.moves[rows,owner,safe,slot]==value.move)&(state.pp[rows,owner,safe,slot]>0)&(state.last_action_kind[rows,owner]==2)&(state.last_action_index[rows,owner]==value.slot)
    good &= j.sum(state.moves[rows,owner,safe]==value.move[:,None],axis=1)==1
    return value.slot,good
