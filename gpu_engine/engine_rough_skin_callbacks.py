"""Original Gen3 Rough Skin contact write, never an HP state-diff producer.

RAW42 carries the actual actor HP before/after assignment. Source admission is
limited to a nonlethal plain contact following its exact registered main hit.
Missing contexts, secondary interleaving, KO and prior errors remain UNKNOWN;
the physical mechanic runs independently of this optional observer.
"""
import hashlib,json,subprocess
from pathlib import Path
from typing import NamedTuple
import jax.numpy as j
import numpy as np
from . import engine_lifecycle_callbacks as life,engine_public_events as public

ROUGH_SKIN_HP=42

class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    candidates:object
    rough_skin:object

class Pending(NamedTuple):
    context:object
    captured:object
    ticket:object
    actor:object
    physical:object
    target:object
    move:object
    old_hp:object
    maximum:object
    status:object
    requested:object
    supported:object

def compile_data(catalog,metadata,rules,ordinary):
    from . import engine_request_source as request
    from .engine_private_team import engine_metadata_binding
    origin=request._rules_origin(rules)
    if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding) or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):
        raise ValueError('same original Rough Skin metadata/rules required')
    script=r"""
const fs=require('fs'),path=require('path'),root=process.argv[1],m=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,m),{Dex}=require(path.join(root,'dist/sim/dex.js')),d=Dex.mod('gen3');
process.stdout.write(JSON.stringify({handler:d.abilities.get('roughskin').onDamagingHit.toString(),contact:m.vocabulary.moves.map(id=>!!d.moves.get(id).flags.contact),source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));
"""
    run=subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30)
    descriptor=json.loads(run.stdout)
    if json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:
        raise ValueError('original Rough Skin source changed')
    handler=''.join(descriptor['handler'].split()).replace('"',"'")
    if handler!="onDamagingHit(damage,target,source,move){if(damage&&move.flags['contact']){this.damage(source.baseMaxhp/16,source,target);}}":
        raise ValueError('exact original Gen3 Rough Skin handler required')
    table=catalog.engine_table.secondary
    if table is None:raise ValueError('original contact setter required')
    rough=np.array([x=='roughskin' for x in metadata['vocabulary']['abilities']],bool)
    if not np.array_equal(rough,np.asarray(table.rough_skin)):raise ValueError('original Rough Skin flags required')
    # Reuse the original source-classified ordinary positive-power family;
    # this permits plain target secondary metadata, not its actual proc span.
    mask=np.asarray(ordinary.candidates)&np.asarray(table.contact)&np.asarray(descriptor['contact'],bool)
    payload=dict(descriptor=descriptor,candidates=mask.tolist(),rough_skin=rough.tolist(),
        engine=catalog.engine_binding,source=catalog.source_binding,producer=catalog.binding,
        helper=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    words=lambda x:j.asarray(x,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),
        words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),
        words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(mask),j.asarray(rough))

def shape(data):
    if type(data) is not Data:raise ValueError('typed original Rough Skin data required')
    for x in data[:5]:life._typed(x,(8,),j.uint32)
    for x in data[5:]:life._typed(x,(len(x),),j.bool_)

def begin(tape,context,captured,state,side,move,table,*,requested):
    from . import engine_execution_completeness as ex,engine_direct_hit_controls as direct
    from . import engine_kernel_frame_risk as risks
    b=len(tape.count);rows=j.arange(b);actor=j.broadcast_to(side,(b,));physical=state.active[rows,actor];target=state.active[rows,1-actor]
    supported=j.zeros(b,bool)
    if context is not None and context.data.rough_skin is not None and type(captured) is risks.CapturedDirectHit:
        data=context.data.rough_skin;shape(data)
        supported=direct._registered_bound(tape,context)
        for name in ('engine_binding','source_binding','move_binding','producer_binding'):supported &= j.all(getattr(data,name)==getattr(tape,name))
        for actual,expected in ((table.contact,data.candidates),(table.rough_skin,data.rough_skin)):
            # Candidate rows are a subset of installed contact flags.
            life._typed(actual,expected.shape,j.bool_)
        supported &= j.all(table.rough_skin==data.rough_skin)
        supported &= ((move>0)&(move<len(data.candidates))&data.candidates[j.clip(move,0,len(data.candidates)-1)])
        holder=state.ability[rows,1-actor,target]
        supported &= (holder>=0)&(holder<len(data.rough_skin))&data.rough_skin[j.clip(holder,0,len(data.rough_skin)-1)]
        supported &= ((captured.actor==actor)&(captured.actor_physical==physical)&(captured.target_physical==target)&(captured.move==move))
        actual=ex._context(tape)
        if actual is not None:
            slot,same=ex._ticket(actual.ledger,captured.ticket)
            supported &= (same&(actual.ledger.stage[rows,slot]==2)&(actual.ledger.end[rows,slot]==tape.count)
                &ex._direct_hit_span_matches(tape,captured.ticket,actor,physical,target,move,captured.write,data=context.data))
            supported &= ((actual.ledger.refusal==0)&((tape.refusal&~public.UNINSTRUMENTED)==0))
        else:supported &= False
    tape,ticket=ex.open_attempt(tape,ex.Site.ROUGH_SKIN_HP,actor,active=requested)
    return tape,Pending(context,captured,ticket,actor,physical,target,move,state.hp[rows,actor,physical],
        state.max_hp[rows,actor,physical],state.status[rows,actor,physical],requested,supported)

def valid_raw(raw):
    return ((raw[:,0]==ROUGH_SKIN_HP)&(raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,2]>=0)&(raw[:,2]<6)
        &(raw[:,3]==1-raw[:,1])&(raw[:,4]>=0)&(raw[:,4]<6)&(raw[:,5]>0)
        &(raw[:,6]>raw[:,7])&(raw[:,7]>0)&(raw[:,8]>=raw[:,6])&(raw[:,8]<=1_000_000)
        &(raw[:,6]-raw[:,7]==j.maximum(1,raw[:,8]//16))&(raw[:,9]>=0)&(raw[:,9]<=6))

def record(tape,p,state):
    from . import engine_execution_completeness as ex,engine_status_callbacks as status
    b=len(tape.count);rows=j.arange(b);hp=state.hp[rows,p.actor,p.physical]
    raw=j.stack((j.full_like(p.actor,ROUGH_SKIN_HP),p.actor,p.physical,1-p.actor,p.target,p.move,p.old_hp,hp,p.maximum,p.status),axis=1)
    good=p.supported&valid_raw(raw)&(state.active[rows,p.actor]==p.physical)&(state.active[rows,1-p.actor]==p.target)
    good &= (state.max_hp[rows,p.actor,p.physical]==p.maximum)&(state.status[rows,p.actor,p.physical]==p.status)
    tape=public._append(tape,ROUGH_SKIN_HP,p.actor,p.physical,1-p.actor,p.target,tuple(raw[:,i]for i in range(5,10)),p.requested&good)
    actual=ex._context(tape)
    if actual is None:return status.unknown(tape,p.requested)
    ledger=ex.unhandled(actual.ledger,p.ticket,active=p.requested&~good,tables=actual.tables)
    ledger=ex.emitted(ledger,p.ticket,tape,active=p.requested&good,tables=actual.tables)
    return ex._put_context(tape,ledger,actual)

def owned(tape,index,cursor):
    from . import engine_execution_completeness as ex
    actual=ex._context(tape)
    if actual is None or actual.secondary_status is None or actual.secondary_status.data.rough_skin is None:return j.zeros_like(tape.count,bool)
    data=actual.secondary_status.data.rough_skin;shape(data)
    raw=tape.words[:,index];b=len(tape.count);rows=j.arange(b);claim=actual.ledger.claimed[:,index]
    ledger=actual.ledger;slot=j.clip(claim-1,0,ledger.site.shape[1]-1);actor=j.clip(raw[:,1],0,1)
    from . import engine_direct_hit_controls as direct
    good=(direct._registered_bound(tape,actual.secondary_status)&ex._validate(ledger,actual.tables)&ex._tape(ledger,tape,actual.tables)&valid_raw(raw)
        &life._valid(cursor)&(claim>0)&(claim<=ledger.count)&(ledger.site[rows,slot]==int(ex.Site.ROUGH_SKIN_HP))
        &(ledger.stage[rows,slot]==2)&(ledger.side[rows,slot]==raw[:,1])&(ledger.start[rows,slot]==index)&(ledger.end[rows,slot]==index+1)
        &(cursor.physical_active[rows,actor]==raw[:,2])&(cursor.physical_active[rows,1-actor]==raw[:,4])
        &(cursor.turn==ledger.turn[rows,slot])&(cursor.phase==ledger.phase[rows,slot])&j.all(cursor.generation==ledger.generation[rows,slot],axis=1)
        &(raw[:,5]<len(data.candidates))&data.candidates[j.clip(raw[:,5],0,len(data.candidates)-1)]
        &j.all(ledger.words[:,index]==raw,axis=1)&(tape.event_turn[:,index]==cursor.turn)&(tape.event_phase[:,index]==cursor.phase)
        &j.all(tape.event_generation[:,index]==cursor.generation,axis=1))
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):good &= j.all(getattr(data,name)==getattr(tape,name))
    return good
