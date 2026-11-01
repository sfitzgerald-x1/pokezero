"""Literal singleton Gen3 Focus Punch preparation, no numerical implementation.

RAW59=[59,side,physical,move,oldBits,newBits,slot,PP,HP,status].
Only the original BEFORE-turn singleton assignment/annotation is observed.
Lost focus, called/queued requests and two preparation handlers stay UNKNOWN.
"""
import hashlib,json,subprocess
from pathlib import Path
from typing import NamedTuple
import jax.numpy as j
import numpy as np
from . import engine_public_events as public,engine_lifecycle_callbacks as life
from .state import FOCUS_PUNCH,FOCUS_LOST,BattleState
FOCUS_PREPARATION=59
SOURCE={'beforeTurnCallback':'545f74252c2c68e0eef833663a1a64997c2f424ed8b102485ed09808d0240d7b','onTry':'946d343b066b9707934dce1dae8af4e76b237dca122b5abccda4d56f4b3788cd','onStart':'ae8036a25cda730a664cee2a4a9224672325069ec9bf60091c318f0e4a02e081','onHit':'013cd36a491f0b298d51966bfec66f7d15fc2fff6efc792527feaa92a7492b0f','onTryAddVolatile':'2f36c3936e46438f551b78354bc00c693f30d62c78d9201e73e63d20aefb7552'}
class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    move:object
    abilities:object
    installed:object
class Pending(NamedTuple):
    context:object
    ticket:object
    before:object
    side:object
    physical:object
    move:object
    slot:object
    requested:object
    valid:object
class Captured(NamedTuple):
    pending:object
    raw:object
    after:object

def compile_data(catalog,metadata,rules):
    from . import engine_request_source as request
    from .engine_private_team import engine_metadata_binding
    origin=request._rules_origin(rules)
    if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding)or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):raise ValueError('same original Focus Punch metadata/rules required')
    script=r"""
const fs=require('fs'),path=require('path'),crypto=require('crypto'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3'),m=d.moves.get('focuspunch'),c=d.conditions.get('focuspunch');
const hash=f=>crypto.createHash('sha256').update(String(f)).digest('hex');
const source={beforeTurnCallback:hash(m.beforeTurnCallback),onTry:hash(m.onTry),onStart:hash(c.onStart),onHit:hash(c.onHit),onTryAddVolatile:hash(c.onTryAddVolatile)};
const abilities=meta.vocabulary.abilities.map(id=>{const a=d.abilities.get(id);return {id:a.id,exists:a.exists,gen:a.gen,tryAdd:a.onTryAddVolatile!==undefined}});
process.stdout.write(JSON.stringify({id:m.id,exists:m.exists,gen:m.gen,power:m.basePower,type:m.type,category:m.category,target:m.target,accuracy:m.accuracy,priority:m.priority,callbacks:Object.keys(m).filter(k=>(/^on/.test(k)||/Callback$/.test(k))&&m[k]!==undefined).sort(),condition:Object.keys(c).filter(k=>/^on/.test(k)&&c[k]!==undefined).sort(),duration:c.duration,source,abilities,source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));"""
    row=json.loads(subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30).stdout)
    if row['source']!=SOURCE or row['id']!='focuspunch'or not row['exists']or row['gen']!=3 or row['power']!=150 or row['type']!='Fighting'or row['category']!='Physical'or row['target']!='normal'or row['accuracy']!=100 or row['priority']!=-3 or row['callbacks']!=['beforeTurnCallback','onTry']or row['condition']!=['onHit','onStart','onTryAddVolatile']or row['duration']!=1:raise ValueError('original Gen3 Focus Punch source differs')
    if json.dumps(row['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(row['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original Focus Punch source changed')
    t=catalog.engine_table;installed=t.focus_punch
    if installed is None:return None
    names=metadata['vocabulary']['moves'];mask=np.asarray([n=='focuspunch'for n in names],bool)
    if not np.array_equal(installed.move,mask)or not np.array_equal(installed.damaging,np.asarray([bool(r and r.get('category','Status')!='Status')for r in metadata['moves']],bool)):raise ValueError('original installed Focus Punch flags required')
    indices=np.flatnonzero(mask);move=int(indices[0])if len(indices)==1 else 0
    if move and not(catalog.ids[move]>0 and catalog.targets[move]==1 and int(t.power[move])==150 and bool(t.physical[move])and int(t.priority[move])==-3 and int(t.accuracy[move])==100 and not bool(t.always_hit[move])and int(t.type[move])==metadata['vocabulary']['types'].index('fighting')):raise ValueError('original Focus Punch numerical row differs')
    abilities=np.asarray([bool(i and r['exists']and r['gen']<=3 and r['id']==metadata['vocabulary']['abilities'][i]and not r['tryAdd'])for i,r in enumerate(row['abilities'])],bool)
    payload=dict(schema='pokezero.literal-singleton-focus-preparation.v1',descriptor=row,move=move,abilities=abilities.tolist(),engine=catalog.engine_binding,source=catalog.source_binding,moves=catalog.move_binding,producer=catalog.binding)
    words=lambda x:j.asarray(x,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(move,j.int32),j.asarray(abilities),installed)

def shape(d):
    if type(d)is not Data:raise ValueError('typed original Focus Punch capability required')
    for v in d[:5]:life._typed(v,(8,),j.uint32)
    life._typed(d.move,(),j.int32);life._typed(d.abilities,(len(d.abilities),),j.bool_)
    m=len(d.installed.move);life._typed(d.installed.move,(m,),j.bool_);life._typed(d.installed.damaging,(m,),j.bool_)

def _bound(tape,context):
    from . import engine_direct_hit_controls as direct
    d=context.data.focus_punch;shape(d);good=direct._registered_bound(tape,context)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);good &= j.all(getattr(d,name)==getattr(tape,name))
    c=tape.execution
    if c is None or c.risks is None:return j.zeros(len(tape.count),bool)
    t=c.risks.tables
    for name,v in (('binding',d.binding),('move',d.move),('abilities',d.abilities)):
        actual=getattr(t,'focus_'+name);life._typed(actual,v.shape,v.dtype);good &= j.all(actual==v)
    return good&j.any(d.binding!=0)&life._valid(tape.cursor)&tape.cursor.initialized&(tape.turn==tape.cursor.turn)&(tape.cursor.phase==0)

def begin(tape,state,moves,slots,context,table,*,ordinary,move_requests):
    from . import engine_execution_completeness as ex
    if context is None or context.data.focus_punch is None:return tape,None
    if type(state)is not BattleState:raise ValueError('actual preparation BattleState required')
    b=len(tape.count);rows=j.arange(b);d=context.data.focus_punch;good=_bound(tape,context)
    life._typed(moves,(b,2),j.int32);life._typed(slots,(b,2),j.int32);life._typed(ordinary,(b,),j.bool_);life._typed(move_requests,(b,2),j.bool_)
    if type(table)is not type(d.installed):raise ValueError('original Focus Punch table required')
    for a,z in zip(table,d.installed,strict=True):
        life._typed(a,z.shape,z.dtype);good &= j.all(a==z)
    hp=state.hp[rows[:,None],j.arange(2)[None,:],state.active]
    charges=move_requests&table.move[j.clip(moves,0,len(table.move)-1)]&(moves>0)&(moves<len(table.move))&(hp>0)
    side=j.argmax(charges,axis=1).astype(j.int32);physical=state.active[rows,side];move=moves[rows,side];slot=slots[rows,side];at=(rows,side,physical);safe=j.clip(slot,0,3)
    requested=ordinary&j.any(charges,axis=1)
    ability=state.ability[at];good &= (j.sum(charges,axis=1)==1)&(move==d.move)&(d.move>0)&(slot>=0)&(slot<4)&(state.moves[at+(safe,)]==move)&(state.pp[at+(safe,)]>0)
    good &= (ability>=0)&(ability<len(d.abilities))&d.abilities[j.clip(ability,0,len(d.abilities)-1)]&(state.base_ability[at]==ability)
    good &= j.all(state.active==tape.cursor.physical_active,axis=1)&j.all(state.volatile==0,axis=(1,2))&~state.pending_residual&~state.done&j.all(~state.forced_switch,axis=1)&j.all(~state.entry_pending,axis=1)&j.all(state.charge_move==0,axis=(1,2))
    good &= (tape.count==1)&((tape.refusal&~public.UNINSTRUMENTED)==0)
    tape,ticket=ex.open_attempt(tape,ex.Site.FOCUS_PREPARATION,side,active=requested)
    return tape,Pending(context,ticket,state,side,physical,move,slot,requested,good)

def _raw(p):
    rows=j.arange(len(p.side));at=(rows,p.side,p.physical);s=p.before;pp=s.pp[at+(j.clip(p.slot,0,3),)]
    return j.stack((j.full_like(p.move,FOCUS_PREPARATION),p.side,p.physical,p.move,j.zeros_like(p.move),j.full_like(p.move,FOCUS_PUNCH),p.slot,pp,s.hp[at],s.status[at]),axis=1)

def owned(tape,index,*,stage=2):
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=j.arange(b);at=j.broadcast_to(j.asarray(index,j.int32),(b,));safe=j.clip(at,0,tape.words.shape[1]-1);raw=tape.words[rows,safe]
    if c is None or c.risks is None:return j.zeros(b,bool)
    t=c.risks.tables
    if t.focus_binding is None:return j.zeros(b,bool)
    life._typed(t.focus_binding,(8,),j.uint32);life._typed(t.focus_move,(),j.int32);life._typed(t.focus_abilities,(len(t.focus_abilities),),j.bool_)
    ledger=c.ledger;good=ex._validate(ledger,c.tables)&ex._tape(ledger,tape,c.tables)&ledger.opened&(ledger.frame>0)&j.any(t.focus_binding!=0)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);life._typed(getattr(t,name),(8,),j.uint32);good &= j.all(getattr(t,name)==getattr(tape,name))
    claim=ledger.count if stage==1 else ledger.claimed[rows,safe];slot=j.clip(claim-1,0,ledger.site.shape[1]-1)
    good &= (at==1)&(at<tape.count)&(raw[:,0]==FOCUS_PREPARATION)&(raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,3]==t.focus_move)&(t.focus_move>0)&(raw[:,4]==0)&(raw[:,5]==FOCUS_PUNCH)&(raw[:,6]>=0)&(raw[:,6]<4)&(raw[:,7]>0)&(raw[:,8]>0)&(raw[:,9]>=0)&(raw[:,9]<=6)
    good &= (claim>0)&(claim<=ledger.count)&(ledger.site[rows,slot]==int(ex.Site.FOCUS_PREPARATION))&(ledger.side[rows,slot]==raw[:,1])&(ledger.stage[rows,slot]==stage)&(ledger.start[rows,slot]==at)&(ledger.no_output_kind[rows,slot]==0)&j.all(tape.switch_details[rows,safe]==0,axis=1)
    good &= j.where(stage==1,tape.count==at+1,ledger.end[rows,slot]==at+1)
    good &= (tape.event_turn[rows,safe]==ledger.turn[rows,slot])&(tape.event_phase[rows,safe]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,safe]==ledger.generation[rows,slot],axis=1)
    prefix=tape.words[:,0];prefix_claim=ledger.claimed[:,0];prefix_slot=j.clip(prefix_claim-1,0,ledger.site.shape[1]-1)
    good &= (prefix[:,0]==life.TURNLOOP_PREFIX)&j.all(prefix[:,1:5]==-1,axis=1)&j.all(prefix[:,5:]==0,axis=1)&(prefix_claim>0)&(ledger.site[rows,prefix_slot]==int(ex.Site.TURNLOOP_PREFIX))&(ledger.stage[rows,prefix_slot]==2)&(ledger.start[rows,prefix_slot]==0)&(ledger.end[rows,prefix_slot]==1)
    good &= (tape.event_turn[:,0]==tape.event_turn[rows,safe])&(tape.event_phase[:,0]==tape.event_phase[rows,safe])&j.all(tape.event_generation[:,0]==tape.event_generation[rows,safe],axis=1)
    if stage==2:good &= (ledger.claimed[rows,safe]==claim)&j.all(ledger.words[rows,safe]==raw,axis=1)&(ledger.event_turn[rows,safe]==tape.event_turn[rows,safe])&(ledger.event_phase[rows,safe]==tape.event_phase[rows,safe])&j.all(ledger.event_generation[rows,safe]==tape.event_generation[rows,safe],axis=1)
    return good

def _available_before(p,tape):
    """Re-read the SAME available operands; a saved valid bit is insufficient."""
    d=p.context.data.focus_punch;s=p.before;b=len(p.side);rows=j.arange(b)
    if type(s)is not BattleState:raise ValueError('actual pre-preparation BattleState required')
    seat=j.clip(p.side,0,1);physical=j.clip(p.physical,0,5);at=(rows,seat,physical);slot=j.clip(p.slot,0,3);ability=s.ability[at]
    good=(p.side>=0)&(p.side<2)&(p.physical>=0)&(p.physical<6)&(s.active[rows,seat]==p.physical)&(p.move==d.move)&(d.move>0)&(p.slot>=0)&(p.slot<4)&(s.moves[at+(slot,)]==p.move)&(s.pp[at+(slot,)]>0)
    good &= (ability>=0)&(ability<len(d.abilities))&d.abilities[j.clip(ability,0,len(d.abilities)-1)]&(s.base_ability[at]==ability)&(s.hp[at]>0)
    good &= j.all(s.active==tape.cursor.physical_active,axis=1)&j.all(s.volatile==0,axis=(1,2))&~s.pending_residual&~s.done&j.all(~s.forced_switch,axis=1)&j.all(~s.entry_pending,axis=1)&j.all(s.charge_move==0,axis=(1,2))
    return good

def _same_write(p,state):
    if type(state)is not BattleState:raise ValueError('actual after-preparation BattleState required')
    b=len(p.side);rows=j.arange(b);expected=p.before.volatile.at[rows,p.side,p.physical].set(j.uint32(FOCUS_PUNCH));good=j.all(state.volatile==expected,axis=(1,2))
    for name in BattleState._fields:
        if name=='volatile':continue
        a,z=getattr(p.before,name),getattr(state,name);life._typed(z,a.shape,a.dtype);equal=a==z;good &= j.all(equal,axis=tuple(range(1,equal.ndim)))if equal.ndim>1 else equal
    return good

def finish(tape,p,state):
    if p is None:return tape,None
    from . import engine_execution_completeness as ex
    if type(state)is not BattleState:raise ValueError('actual after-preparation BattleState required')
    good=p.valid&_bound(tape,p.context)&_available_before(p,tape)&_same_write(p,state)
    raw=_raw(p);use=p.requested&good;before=tape.count
    tape=public._append(tape,FOCUS_PREPARATION,p.side,p.physical,p.move,j.zeros_like(p.move),tuple(raw[:,i]for i in range(5,10)),use)
    valid=good&owned(tape,before,stage=1);_,same=ex._ticket(tape.execution.ledger,p.ticket);valid &= same
    c=ex._context(tape);ledger=ex.unhandled(c.ledger,p.ticket,active=p.requested&~valid,tables=c.tables);ledger=ex.emitted(ledger,p.ticket,tape,active=p.requested&valid,tables=c.tables);tape=ex._put_context(tape,ledger,c)
    tape=tape._replace(refusal=tape.refusal|j.where(p.requested&~valid,256,0))
    return tape,Captured(p,raw,state)

def matches_selected(tape,c,state,side,move,caller_move):
    if c is None:return j.zeros(len(tape.count),bool)
    if type(c)is not Captured or type(c.pending)is not Pending or type(state)is not BattleState:raise ValueError('typed SAME Focus Punch preparation/current state required')
    from . import engine_execution_completeness as ex
    p=c.pending;b=len(tape.count);rows=j.arange(b)
    for x in (side,move,caller_move):life._typed(x,(b,),j.int32)
    slot,same=ex._ticket(tape.execution.ledger,p.ticket);ledger=tape.execution.ledger;at=ledger.start[rows,slot];raw=tape.words[rows,j.clip(at,0,tape.words.shape[1]-1)]
    good=same&p.requested&p.valid&_bound(tape,p.context)&_available_before(p,tape)&_same_write(p,c.after)&owned(tape,at)&j.all(raw==c.raw,axis=1)&j.all(raw==_raw(p),axis=1)
    good &= (side==p.side)&(move==p.move)&(caller_move==p.move)&(state.active[rows,j.clip(side,0,1)]==p.physical)&(state.hp[rows,p.side,p.physical]>0)&(state.volatile[rows,p.side,p.physical]==FOCUS_PUNCH)&(state.confusion_counter[rows,p.side,p.physical]==0)
    # The actual ordinary MOVE is the current declaration, following this
    # preparation exactly once; a trailing/called/foreign header cannot borrow it.
    current=j.clip(tape.count-1,0,tape.words.shape[1]-1);decl=tape.words[rows,current]
    good &= (tape.count>at+1)&(decl[:,0]==public.MOVE)&(decl[:,1]==side)&(decl[:,2]==p.physical)&(decl[:,3]==1-side)&(decl[:,4]==state.active[rows,1-p.side])&(decl[:,5]==move)&j.all(decl[:,6:]==0,axis=1)
    present=j.arange(tape.words.shape[1])[None,:]<tape.count[:,None]
    same_move=(tape.words[:,:,0]==public.MOVE)&(tape.words[:,:,1]==p.side[:,None])&(tape.words[:,:,5]==p.move[:,None])
    good &= j.sum(present&same_move,axis=1)==1
    good &= (tape.event_turn[rows,current]==ledger.turn[rows,slot])&(tape.event_phase[rows,current]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,current]==ledger.generation[rows,slot],axis=1)
    good &= (ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)&j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)&(p.physical==tape.cursor.physical_active[rows,p.side])
    return good
