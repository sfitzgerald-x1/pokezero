"""Literal plain move-secondary status writes; no chance/physics reconstruction.

RAW9 is the sole public status line. RAW36 closes its exact same-setter
actor/move/proc/changed/counter receipt before AfterSetStatus reflection/cures.
Copies are trusted device transport; the owning Prepared guard checks originals.
"""
import hashlib
import json
from pathlib import Path
import subprocess
from typing import NamedTuple

import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import normalize_id
from . import engine_lifecycle_callbacks as life, engine_public_events as public, mechanics as m

SECONDARY_STATUS=36
ELIGIBLE,TRIGGER,PROC,CHANGED,LIVE,SUBSTITUTE,DUST=(1<<i for i in range(7))
REQUIRED=ELIGIBLE|TRIGGER|PROC|CHANGED|LIVE

class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    candidates:object
    status:object
    chance:object
    synchronize:object
    lum:object
    installed:object

class Write(NamedTuple):
    eligible:object
    trigger:object
    proc:object
    changed:object
    live:object
    substitute:object
    dust:object
    old_status:object
    new_status:object
    old_counter:object
    new_counter:object
    actor_ability:object
    target_ability:object
    target_item:object

def compile_data(catalog,metadata,rules):
    from . import engine_request_source as request
    from .engine_private_team import engine_metadata_binding
    origin=request._rules_origin(rules)
    if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding) or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):
        raise ValueError('same original secondary status metadata/rules required')
    script=r"""
const fs=require('fs'),path=require('path'),root=process.argv[1];const meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta);const {Dex}=require(path.join(root,'dist/sim/dex.js'));const d=Dex.mod('gen3');
const rows=meta.vocabulary.moves.map(id=>{const m=d.moves.get(id);return {id:m.id,exists:m.exists,gen:m.gen,
target:m.target,power:m.basePower,category:m.category,contact:!!m.flags.contact,secondary:m.secondary||null,
callbacks:Object.keys(m).filter(k=>/^on/.test(k)&&m[k]!==undefined),mixed:!!((m.secondaries&&(!Array.isArray(m.secondaries)||m.secondaries.length!==1||JSON.stringify(m.secondaries[0])!==JSON.stringify(m.secondary)))||m.self||m.status||m.boosts||m.volatileStatus||m.sideCondition||m.slotCondition||m.condition||m.weather||m.terrain||m.pseudoWeather||m.recoil||m.drain||m.heal||m.selfdestruct||m.forceSwitch||m.selfSwitch)};});
process.stdout.write(JSON.stringify({rows,source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));
"""
    run=subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30)
    descriptor=json.loads(run.stdout)
    if json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:
        raise ValueError('original secondary source changed')
    t=catalog.engine_table;s=t.secondary
    if s is None:raise ValueError('installed original secondary setter required')
    codes={'par':m.PAR,'psn':m.PSN,'brn':m.BRN,'frz':m.FRZ};mask=np.zeros(len(t.power),bool)
    for i,(name,row) in enumerate(zip(metadata['vocabulary']['moves'],descriptor['rows'],strict=True)):
        secondary=row['secondary']
        if not isinstance(secondary,dict):continue
        status=codes.get(secondary.get('status'),0);chance=secondary.get('chance')
        mask[i]=bool(i>0 and row['exists'] and row['gen']<=3 and normalize_id(row['id'])==normalize_id(name)
            and row['target']=='normal' and row['power']>0 and row['category'] in ('Physical','Special')
            and not row['callbacks'] and not row['mixed'] and set(secondary)=={'chance','status'}
            and status and type(chance) is int and 0<chance<=100
            and int(t.power[i])==row['power'] and int(s.status[i])==status and int(s.chance[i])==chance
            and bool(s.contact[i])==row['contact']
            and not any(bool(x[i]) for x in (s.spd_drop,s.flinch,s.speed_drop,s.self_omni))
            and catalog.ids[i]>0 and catalog.targets[i]==1)
    if not mask.any():raise ValueError('no installed original plain secondary rows')
    payload=dict(schema='pokezero.secondary-status.literal.v1',descriptor=descriptor,mask=mask.tolist(),
        source=catalog.source_binding,engine=catalog.engine_binding,producer=catalog.binding,
        installed={n:np.asarray(x).tolist()for n,x in zip(s._fields,s)})
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest()
    words=lambda x:j.asarray(x,j.uint32)
    lum=np.asarray(t.items.lum) if t.items is not None else np.zeros(len(metadata['vocabulary']['items']),bool)
    return Data(words(np.frombuffer(digest,dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),
        words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(mask),s.status,s.chance,s.synchronize,j.asarray(lum),s)

def shape(data):
    if type(data) is not Data:raise ValueError('typed original secondary data required')
    for x in data[:5]:life._typed(x,(8,),j.uint32)
    n=len(data.candidates)
    for x,dtype in ((data.candidates,j.bool_),(data.status,j.int32),(data.chance,j.int32)):life._typed(x,(n,),dtype)
    for x in (data.synchronize,data.lum):life._typed(x,(len(x),),j.bool_)

def _bound(tape,context,table):
    from . import engine_direct_hit_controls as direct,engine_execution_completeness as ex
    data=context.data.secondary_status
    shape(data)
    bound=direct._registered_bound(tape,context)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):bound &= j.all(getattr(data,name)==getattr(tape,name))
    # The SAME installed setter arrays are used by this actual call.
    if type(table) is not type(data.installed):raise ValueError('original installed secondary type required')
    for actual,expected in zip(table,data.installed,strict=True):
        if actual.shape!=expected.shape or actual.dtype!=expected.dtype:raise ValueError('installed secondary shape/dtype changed')
        bound &= j.all(actual==expected)
    return bound

def begin(tape,context,table,side,physical,target,move,write):
    from . import engine_execution_completeness as ex
    if context is None or context.data.secondary_status is None:return tape,None
    _bound(tape,context,table)
    b=len(tape.count)
    for x in write[:7]:life._typed(x,(b,),j.bool_)
    for x in (*write[7:],side,physical,target,move):life._typed(x,(b,),j.int32)
    actual=ex._context(tape);tape=tape._replace(execution=actual._replace(secondary_status=context))
    requested=write.changed
    tape,ticket=ex.open_attempt(tape,ex.Site.SECONDARY_STATUS,1-side,active=requested)
    return tape,ticket

def raw_words(side,physical,target,move,w):
    flags=j.zeros_like(move)
    for i,value in enumerate(w[:7]):flags |= value.astype(j.int32)<<i
    return j.stack((j.full_like(move,SECONDARY_STATUS),side,physical,1-side,target,move,flags,w.old_status,w.new_status,w.old_counter|(w.new_counter<<4)),axis=1)

def valid_raw(raw,data):
    shape(data);move=raw[:,5];safe=j.clip(move,0,len(data.candidates)-1)
    return ((raw[:,0]==SECONDARY_STATUS)&(raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,3]==1-raw[:,1])
        &(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,4]>=0)&(raw[:,4]<6)&(move>0)&(move<len(data.candidates))
        &data.candidates[safe]&(raw[:,6]==REQUIRED)&(raw[:,7]==m.NONE)&(raw[:,8]==data.status[safe])&(raw[:,9]==0))

def finish(tape,ticket,side,physical,target,move,w,context,table):
    from . import engine_execution_completeness as ex,engine_status_callbacks as status
    if ticket is None:return status.unknown(tape,w.changed)
    data=context.data.secondary_status;raw=raw_words(side,physical,target,move,w)
    abi=j.clip(w.target_ability,0,len(data.synchronize)-1);item=j.clip(w.target_item,0,len(data.lum)-1)
    valid=_bound(tape,context,table)&valid_raw(raw,data)&(w.target_ability>=0)&(w.target_ability<len(data.synchronize))&~data.synchronize[abi]
    valid &= (w.target_item>=0)&(w.target_item<len(data.lum))&~data.lum[item]
    requested=w.changed
    tape=status.changed(tape,1-side,target,w.old_status,w.new_status,status.PRIMARY,eligible=requested&valid)
    tape=public._append(tape,SECONDARY_STATUS,side,physical,1-side,target,tuple(raw[:,i]for i in range(5,10)),requested&valid)
    actual=ex._context(tape);ledger=ex.unhandled(actual.ledger,ticket,active=requested&~valid,tables=actual.tables)
    tape=ex._put_context(tape,ledger,actual)
    # Exact source receipt is rechecked from the actual appended words.
    actual=ex._context(tape);good=owned_pair(tape,j.maximum(tape.count-1,0),stage=1)
    ledger=ex.emitted(actual.ledger,ticket,tape,active=requested&valid&good,tables=actual.tables)
    ledger=ex.unhandled(ledger,ticket,active=requested&~good,tables=actual.tables)
    tape=ex._put_context(tape,ledger,actual)
    return status.unknown(tape,requested&~(valid&good))

def owned_pair(tape,index,*,stage=2):
    from . import engine_execution_completeness as ex,engine_direct_hit_controls as direct
    b=len(tape.count);rows=j.arange(b);context=ex._context(tape)
    if context is None or context.secondary_status is None:return j.zeros(b,bool)
    original=context.secondary_status;data=original.data.secondary_status
    if data is None:return j.zeros(b,bool)
    at=j.broadcast_to(j.asarray(index,j.int32),(b,));first=j.clip(at-1,0,tape.words.shape[1]-1);last=j.clip(at,0,tape.words.shape[1]-1)
    raw=tape.words[rows,last];status=tape.words[rows,first];ledger=context.ledger
    # At stage1 the fresh ticket has not claimed storage yet.
    claim=j.where(stage==1,ledger.count,ledger.claimed[rows,last]);slot=j.clip(claim-1,0,ledger.site.shape[1]-1)
    valid=direct._registered_bound(tape,original)&valid_raw(raw,data)&(at>0)&(at<tape.count)&(claim>0)&(claim<=ledger.count)
    valid &= (ledger.site[rows,slot]==int(ex.Site.SECONDARY_STATUS))&(ledger.side[rows,slot]==raw[:,3])&(ledger.stage[rows,slot]==stage)
    valid &= (ledger.start[rows,slot]==first)&j.where(stage==1,tape.count==at+1,ledger.end[rows,slot]==at+1)
    valid &= (status[:,0]==9)&j.all(status[:,1:3]==-1,axis=1)&j.all(status[:,3:5]==raw[:,3:5],axis=1)
    valid &= (status[:,5]==raw[:,7])&(status[:,6]==raw[:,8])&j.all(status[:,7:]==0,axis=1)
    positions=j.arange(tape.words.shape[1])[None,:]
    declared=j.max(j.where((positions<first[:,None])&(tape.words[:,:,0]==public.MOVE),positions,-1),axis=1)
    source=tape.words[rows,j.maximum(declared,0)]
    valid &= (declared>=0)&j.all(source[:,1:6]==raw[:,1:6],axis=1)&j.all(source[:,6:]==0,axis=1)
    valid &= (tape.event_turn[rows,j.maximum(declared,0)]==ledger.turn[rows,slot])&(tape.event_phase[rows,j.maximum(declared,0)]==ledger.phase[rows,slot])
    valid &= j.all(tape.event_generation[rows,j.maximum(declared,0)]==ledger.generation[rows,slot],axis=1)
    if stage==1:
        valid &= (raw[:,2]==tape.cursor.physical_active[rows,j.clip(raw[:,1],0,1)])&(raw[:,4]==tape.cursor.physical_active[rows,j.clip(raw[:,3],0,1)])
    for where in (first,last):
        valid &= (tape.event_turn[rows,where]==ledger.turn[rows,slot])&(tape.event_phase[rows,where]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,where]==ledger.generation[rows,slot],axis=1)
        if stage==2:
            valid &= (ledger.claimed[rows,where]==claim)&j.all(ledger.words[rows,where]==tape.words[rows,where],axis=1)
    return valid
