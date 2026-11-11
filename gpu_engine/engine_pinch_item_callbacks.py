"""Same-call original residual pinch item/boost transaction (RAW46, Site4).

Only original Gen3 Liechi/Petaya/Salac successful uncapped +1 writes. Neither
private HP nor an inferred final-state transaction is projected publicly.
"""
import hashlib,json,subprocess
from pathlib import Path
from typing import NamedTuple
import jax.numpy as j
import numpy as np
from . import engine_public_events as public,engine_lifecycle_callbacks as life

PINCH_ITEM=46
class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    stats:object
class Pending(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    item:object
    boosts:object
    requested:object
    valid:object

def compile_data(catalog,metadata,rules):
    from . import engine_request_source as request
    from .engine_private_team import engine_metadata_binding
    origin=request._rules_origin(rules)
    if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding) or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):raise ValueError('same original pinch metadata/rules required')
    script=r"""
const fs=require('fs'),path=require('path'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3');
const rows=meta.vocabulary.items.map(id=>{const i=d.items.get(id);return {id:i.id,exists:i.exists,gen:i.gen,
update:i.onUpdate===undefined,residual:i.onResidual?i.onResidual.toString():null,eat:i.onEat?i.onEat.toString():null,
order:i.onResidualOrder,suborder:i.onResidualSubOrder}});
process.stdout.write(JSON.stringify({rows,source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));"""
    p=subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30)
    descriptor=json.loads(p.stdout)
    if json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original pinch source changed')
    norm=lambda s:''.join((s or '').split())
    expected={'liechiberry':(0,'atk'),'petayaberry':(2,'spa'),'salacberry':(4,'spe')}
    stats=np.full(len(descriptor['rows']),-1,np.int32)
    installed=catalog.engine_table.items
    if installed is None:raise ValueError('original installed item table required')
    for i,row in enumerate(descriptor['rows']):
        if row['id'] not in expected:continue
        stat,name=expected[row['id']]
        if not (i>0 and row['id']==metadata['vocabulary']['items'][i] and row['exists'] and row['gen']<=3 and row['update'] and row['order']==10 and row['suborder']==4
            and norm(row['residual'])=='onResidual(pokemon){if(pokemon.hp<=pokemon.maxhp/4){pokemon.eatItem();}}'
            and norm(row['eat'])=='onEat(pokemon){this.boost({'+name+':1});}'
            and int(installed.pinch_stat[i])==stat):raise ValueError('original Gen3 pinch handler/table differs')
        stats[i]=stat
    if not np.array_equal(np.asarray(installed.pinch_stat),stats):raise ValueError('unsupported installed pinch item')
    payload=dict(schema='pokezero.original-residual-pinch.v1',descriptor=descriptor,stats=stats.tolist(),engine=catalog.engine_binding,source=catalog.source_binding,moves=catalog.move_binding,producer=catalog.binding)
    words=lambda v:j.asarray(v,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(stats))

def shape(d):
    if type(d)is not Data:raise ValueError('original typed pinch data required')
    for v in d[:5]:life._typed(v,(8,),j.uint32)
    if d.stats.ndim!=1 or not len(d.stats):raise ValueError('nonempty original item namespace required')
    life._typed(d.stats,d.stats.shape,j.int32)

def _bound(tape,context):
    from . import engine_direct_hit_controls as direct
    d=context.data.pinch_item;shape(d);same=direct._registered_bound(tape,context)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32)
        same &= j.all(getattr(d,name)==getattr(tape,name))
    r=tape.execution.risks
    if r is None:return j.zeros(len(tape.count),bool)
    life._typed(r.tables.pinch_binding,(8,),j.uint32);life._typed(r.tables.pinch_stats,d.stats.shape,j.int32)
    return same&j.any(d.binding!=0)&j.all(r.tables.pinch_binding==d.binding)&j.all(r.tables.pinch_stats==d.stats)

def begin(tape,state,side,item,boosts,hp,maximum,table,context,*,eligible):
    from . import engine_execution_completeness as ex
    if context is None or context.data.pinch_item is None:return tape,None
    d=context.data.pinch_item;shape(d);b=len(tape.count);rows=j.arange(b)
    for x in (side,item,hp,maximum):life._typed(x,(b,),j.int32)
    life._typed(boosts,(b,7),j.int32);life._typed(eligible,(b,),j.bool_)
    life._typed(table.pinch_stat,d.stats.shape,j.int32)
    stat=d.stats[j.clip(item,0,len(d.stats)-1)]
    requested=eligible&(hp>0)&(hp*4<=maximum)&(table.pinch_stat[j.clip(item,0,len(d.stats)-1)]>=0)
    safe=j.clip(side,0,1);physical=state.active[rows,safe]
    good=_bound(tape,context)&j.all(table.pinch_stat==d.stats)&(item>0)&(item<len(d.stats))&(stat>=0)
    good &= (side>=0)&(side<2)&(physical==tape.cursor.physical_active[rows,safe])&tape.cursor.initialized&(tape.cursor.phase==0)&(tape.turn==tape.cursor.turn)
    good &= (maximum>0)&(maximum<=1000000)&(hp<=maximum)&j.all((boosts>=-6)&(boosts<=6),axis=1)&(boosts[rows,j.clip(stat,0,6)]<6)
    tape,ticket=ex.open_attempt(tape,ex.Site.PINCH_ITEM,side,active=requested)
    return tape,Pending(context,ticket,side,physical,item,boosts,requested,good)

def owned(tape,index,*,stage=2):
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=j.arange(b);at=j.broadcast_to(j.asarray(index,j.int32),(b,));pos=j.clip(at,0,tape.words.shape[1]-1);raw=tape.words[rows,pos]
    if c is None or c.risks is None:return j.zeros(b,bool)
    t=c.risks.tables;l=c.ledger
    life._typed(t.pinch_binding,(8,),j.uint32)
    if t.pinch_stats.ndim!=1 or not len(t.pinch_stats):raise ValueError('original pinch namespace required')
    life._typed(t.pinch_stats,t.pinch_stats.shape,j.int32)
    claim=j.where(stage==1,l.count,l.claimed[rows,pos]);slot=j.clip(claim-1,0,l.site.shape[1]-1)
    item=raw[:,5];stat=raw[:,6];safe=j.clip(item,0,len(t.pinch_stats)-1)
    good=ex._validate(l,c.tables)&ex._tape(l,tape,c.tables)&l.opened&(l.frame>0)&j.any(t.pinch_binding!=0)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);life._typed(getattr(t,name),(8,),j.uint32)
        good &= j.all(getattr(t,name)==getattr(tape,name))
    good &= (at>=0)&(at<tape.count)&(raw[:,0]==PINCH_ITEM)&(raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,2]>=0)&(raw[:,2]<6)&j.all(raw[:,1:3]==raw[:,3:5],axis=1)
    good &= (item>0)&(item<len(t.pinch_stats))&(stat>=0)&(stat==t.pinch_stats[safe])&(raw[:,7]>=-6)&(raw[:,7]<6)&(raw[:,8]==raw[:,7]+1)&(raw[:,9]==0)
    good &= (claim>0)&(claim<=l.count)&(l.site[rows,slot]==int(ex.Site.PINCH_ITEM))&(l.side[rows,slot]==raw[:,1])&(l.stage[rows,slot]==stage)&(l.start[rows,slot]==at)
    good &= j.where(stage==1,tape.count==at+1,l.end[rows,slot]==at+1)
    good &= (tape.event_turn[rows,pos]==l.turn[rows,slot])&(tape.event_phase[rows,pos]==l.phase[rows,slot])&j.all(tape.event_generation[rows,pos]==l.generation[rows,slot],axis=1)
    if stage==2:
        good &= (l.claimed[rows,pos]==claim)&j.all(l.words[rows,pos]==raw,axis=1)&(l.event_turn[rows,pos]==tape.event_turn[rows,pos])&(l.event_phase[rows,pos]==tape.event_phase[rows,pos])&j.all(l.event_generation[rows,pos]==tape.event_generation[rows,pos],axis=1)
    return good

def finish(tape,p,item,boosts):
    if p is None:return tape
    from . import engine_execution_completeness as ex
    b=len(tape.count);rows=j.arange(b);life._typed(item,(b,),j.int32);life._typed(boosts,(b,7),j.int32)
    d=p.context.data.pinch_item;stat=d.stats[j.clip(p.item,0,len(d.stats)-1)]
    old=p.boosts[rows,j.clip(stat,0,6)];new=boosts[rows,j.clip(stat,0,6)]
    good=p.valid&_bound(tape,p.context)&(item==0)&(new==old+1)&j.all(boosts==p.boosts+(j.arange(7)==stat[:,None]),axis=1)
    l=tape.execution.ledger;slot,same=ex._ticket(l,p.ticket)
    good &= same&(l.turn[rows,slot]==tape.cursor.turn)&(l.phase[rows,slot]==tape.cursor.phase)&j.all(l.generation[rows,slot]==tape.cursor.generation,axis=1)&(p.physical==tape.cursor.physical_active[rows,j.clip(p.side,0,1)])
    room=(tape.count>=0)&(tape.count<tape.words.shape[1]);pos=j.clip(tape.count,0,tape.words.shape[1]-1)
    good &= ~room|j.all(tape.switch_details[rows,pos]==0,axis=1)
    z=j.zeros(b,j.int32)
    tape=public._append(tape,PINCH_ITEM,p.side,p.physical,p.side,p.physical,(p.item,stat,old,new,z),p.requested&good)
    valid=good&room&owned(tape,j.maximum(tape.count-1,0),stage=1)
    c=ex._context(tape);ledger=ex.unhandled(c.ledger,p.ticket,active=p.requested&~valid,tables=c.tables);ledger=ex.emitted(ledger,p.ticket,tape,active=p.requested&valid,tables=c.tables)
    tape=ex._put_context(tape,ledger,c)
    return tape._replace(refusal=tape.refusal|j.where(p.requested&~valid,256,0))
