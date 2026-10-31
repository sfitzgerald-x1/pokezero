"""Same-call ordinary Substitute HP write, never public private-HP output.

RAW38 is an INTERNAL source receipt. Its original FULL expansion contains only
the original Gen3 effectiveness/critical annotations and Substitute line.
Legacy direct compilers have no capability; unsupported attempts retain their
existing direct-hit/accounting/risk refusals. No sticky bit is removed.
"""
import hashlib
import json
from pathlib import Path
import subprocess
from typing import NamedTuple

import jax.numpy as j
import numpy as np

from . import engine_lifecycle_callbacks as life, engine_public_events as public

SUBSTITUTE_DAMAGE = 38


class Data(NamedTuple):
    binding: object
    engine_binding: object
    source_binding: object
    move_binding: object
    producer_binding: object
    candidates: object


class Write(NamedTuple):
    power: object
    hit: object
    immune: object
    protected: object
    enduring: object
    bypass: object
    absorbed: object
    critical: object
    numerator: object
    denominator: object
    old_sub: object
    new_sub: object
    dealt: object
    damage: object
    old_hp: object
    new_hp: object


class Pending(NamedTuple):
    context: object
    ticket: object
    actor: object
    physical: object
    target: object
    move: object
    requested: object


class Captured(NamedTuple):
    pending: Pending
    write: Write


def compile_data(catalog, metadata, rules):
    from . import engine_request_source as request
    from .engine_private_team import engine_metadata_binding
    origin=request._rules_origin(rules)
    if (not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding)
            or not np.array_equal(rules.device.engine_binding,catalog.engine_binding)):
        raise ValueError('same original Substitute damage namespace/rules required')
    script=r"""
const fs=require('fs'),path=require('path'),root=process.argv[1],m=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,m),{Dex}=require(path.join(root,'dist/sim/dex.js')),d=Dex.mod('gen3');
const sub=d.moves.get('substitute').condition;
const rows=m.vocabulary.moves.map(id=>{const x=d.moves.get(id);return {id:x.id,exists:x.exists,gen:x.gen,target:x.target,power:x.basePower,
 category:x.category,bypass:!!x.flags.bypasssub,callbacks:Object.keys(x).filter(k=>(/^on/.test(k)||/Callback$/.test(k))&&x[k]!==undefined),
 exotic:!!(x.multihit||x.damage||x.recoil||x.drain||x.heal||x.self||x.boosts||x.status||x.volatileStatus||x.sideCondition||x.slotCondition||x.condition||x.weather||x.terrain||x.pseudoWeather||x.selfdestruct||x.forceSwitch||x.selfSwitch||x.ohko)}});
process.stdout.write(JSON.stringify({rows,substitute:{hit:sub.onTryPrimaryHit.toString(),end:sub.onEnd.toString()},source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));
"""
    run=subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],
        input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30)
    source=json.loads(run.stdout)
    if (json.dumps(source['source_tree'],sort_keys=True)!=origin.source_tree_json
            or json.dumps(source['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json):
        raise ValueError('original Substitute source changed')
    table=catalog.engine_table
    mask=np.zeros(len(table.power),bool)
    for i,(name,row) in enumerate(zip(metadata['vocabulary']['moves'],source['rows'],strict=True)):
        mask[i]=bool(i>0 and name==row['id'] and row['exists'] and row['gen']<=3
            and row['target'] in ('normal','allAdjacent','allAdjacentFoes') and row['category'] in ('Physical','Special')
            and row['power']>0 and not row['callbacks'] and not row['exotic'] and not row['bypass']
            and int(table.power[i])==row['power'] and catalog.ids[i]>0 and catalog.targets[i]==1)
    payload=dict(source=source,candidates=mask.tolist(),engine=catalog.engine_binding,
        producer=catalog.binding,helper=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    words=lambda x:j.asarray(x,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True,default=lambda x:np.asarray(x).tolist()).encode()).digest(),dtype='>u4').astype(np.uint32)),
        words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),
        words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(mask))


def shape(data):
    if type(data) is not Data:raise ValueError('typed original Substitute damage data required')
    for x in data[:5]:life._typed(x,(8,),j.uint32)
    life._typed(data.candidates,(len(data.candidates),),j.bool_)


def route(context,move,*,acts,power,hit,immune,protected,enduring,bypass,absorbed,sub_hp):
    if context is None or context.data.substitute_damage is None:return j.zeros_like(acts)
    data=context.data.substitute_damage;shape(data)
    valid=(move>0)&(move<len(data.candidates))
    return (acts&valid&data.candidates[j.clip(move,0,len(data.candidates)-1)]&(power>0)
        &hit&~immune&~protected&~enduring&~bypass&~absorbed&(sub_hp>0))


def valid_write(write,b):
    if type(write) is not Write:raise ValueError('literal Substitute damage Write required')
    for i,x in enumerate(write):life._typed(x,(b,),j.bool_ if 1<=i<=7 else j.int32)
    from .engine_direct_hit_controls import _positive_ratio
    return ((write.power>0)&write.hit&~write.immune&~write.protected&~write.enduring
        &~write.bypass&~write.absorbed&_positive_ratio(write.numerator,write.denominator)
        &(write.old_sub>0)&(write.old_sub<=1_000_000)&(write.damage>0)
        &(write.dealt==j.minimum(write.old_sub,write.damage))&(write.dealt>0)
        &(write.new_sub==write.old_sub-write.dealt)&(write.new_sub>=0)
        &(write.old_hp>0)&(write.new_hp==write.old_hp))


def raw_words(p,write):
    effect=j.where(write.numerator>write.denominator,1,j.where(write.numerator<write.denominator,-1,0))
    return j.stack((j.full_like(p.actor,SUBSTITUTE_DAMAGE),p.actor,p.physical,1-p.actor,p.target,
        p.move,write.old_sub,write.new_sub,write.critical.astype(j.int32),effect),axis=1)


def validate_raw(raw):
    return ((raw[:,0]==SUBSTITUTE_DAMAGE)&(raw[:,1]>=0)&(raw[:,1]<2)
        &(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,3]==1-raw[:,1])&(raw[:,4]>=0)&(raw[:,4]<6)
        &(raw[:,5]>0)&(raw[:,6]>raw[:,7])&(raw[:,7]>=0)&(raw[:,6]<=1_000_000)
        &((raw[:,8]==0)|(raw[:,8]==1))&(raw[:,9]>=-1)&(raw[:,9]<=1))


def span_matches(tape,p,write):
    from . import engine_execution_completeness as ex, engine_direct_hit_controls as direct
    actual=ex._context(tape)
    if actual is None or type(p) is not Pending:return j.zeros_like(tape.count,bool)
    bound=direct._registered_bound(tape,p.context)
    data=p.context.data.substitute_damage
    if data is None:return j.zeros_like(tape.count,bool)
    shape(data);b=len(tape.count);rows=j.arange(b);ledger=actual.ledger;slot,same=ex._ticket(ledger,p.ticket)
    for x in (p.actor,p.physical,p.target,p.move):life._typed(x,(b,),j.int32)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):bound &= j.all(getattr(data,name)==getattr(tape,name))
    start=ledger.start[rows,slot];prefix=j.clip(start-1,0,tape.words.shape[1]-1);at=j.clip(start,0,tape.words.shape[1]-1)
    decl=tape.words[rows,prefix];raw=tape.words[rows,at];safe=j.clip(p.actor,0,1)
    good=(bound&same&valid_write(write,b)&p.requested&(p.move>0)&(p.move<len(data.candidates))
        &data.candidates[j.clip(p.move,0,len(data.candidates)-1)]
        &(ledger.site[rows,slot]==int(ex.Site.SUBSTITUTE_DAMAGE))&(ledger.side[rows,slot]==1-p.actor)
        &(start>0)&(tape.count==start+1)&validate_raw(raw)&j.all(raw==raw_words(p,write),axis=1)
        &(decl[:,0]==public.MOVE)&(decl[:,1]==p.actor)&(decl[:,2]==p.physical)
        &(decl[:,3]==1-p.actor)&(decl[:,4]==p.target)&(decl[:,5]==p.move)&j.all(decl[:,6:]==0,axis=1)
        &(p.physical==tape.cursor.physical_active[rows,safe])&(p.target==tape.cursor.physical_active[rows,1-safe])
        &(tape.cursor.turn==ledger.turn[rows,slot])&(tape.cursor.phase==ledger.phase[rows,slot])
        &j.all(tape.cursor.generation==ledger.generation[rows,slot],axis=1))
    for index in (prefix,at):
        good &= ((tape.event_turn[rows,index]==ledger.turn[rows,slot])
            &(tape.event_phase[rows,index]==ledger.phase[rows,slot])
            &j.all(tape.event_generation[rows,index]==ledger.generation[rows,slot],axis=1))
    return good


def record(tape,p,write):
    from . import engine_execution_completeness as ex
    raw=raw_words(p,write);before=tape
    tape=public._append(tape,SUBSTITUTE_DAMAGE,p.actor,p.physical,1-p.actor,p.target,
        tuple(raw[:,i]for i in range(5,10)),p.requested&valid_write(write,len(tape.count)))
    actual=ex._context(tape)
    if actual is None:return before
    good=span_matches(tape,p,write)
    ledger=ex.unhandled(actual.ledger,p.ticket,active=p.requested&~good,tables=actual.tables)
    ledger=ex.emitted(ledger,p.ticket,tape,active=p.requested&good,tables=actual.tables)
    return ex._put_context(tape,ledger,actual)


def owned_raw(tape,index,raw,cursor=None):
    """Read-only stage2 source receipt check; never grants consumer credit."""
    from . import engine_execution_completeness as ex
    actual=ex._context(tape)
    if actual is None:return j.zeros_like(tape.count,bool)
    current=tape.cursor if cursor is None else cursor
    ledger=actual.ledger;b=len(tape.count);rows=j.arange(b);claim=ledger.claimed[:,index];slot=j.clip(claim-1,0,ledger.site.shape[1]-1)
    same=(ex._validate(ledger,actual.tables)&ex._tape(ledger,tape,actual.tables)&validate_raw(raw)
        &(claim>0)&(claim<=ledger.count)&(ledger.site[rows,slot]==int(ex.Site.SUBSTITUTE_DAMAGE))
        &(ledger.side[rows,slot]==raw[:,3])&(ledger.stage[rows,slot]==2)
        &(ledger.start[rows,slot]==index)&(ledger.end[rows,slot]==index+1)
        &j.all(ledger.words[:,index]==raw,axis=1))
    for name in ('turn','phase'):
        same &= (getattr(current,name)==getattr(ledger,name)[rows,slot])
        same &= (getattr(tape,'event_'+name)[:,index]==getattr(ledger,name)[rows,slot])
    same &= j.all(current.generation==ledger.generation[rows,slot],axis=1)
    same &= j.all(tape.event_generation[:,index]==ledger.generation[rows,slot],axis=1)
    return same
