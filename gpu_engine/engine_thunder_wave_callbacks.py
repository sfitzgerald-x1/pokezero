"""Actual canonical Thunder Wave NONE->PAR input; existing RAW9+26 only.

The status write is made by the original numerical kernel. This observer emits
its literal guard continuation, then requires the same current MOVE/status
span and both original consumers. Explicit SAME source rules additionally bind
only the actual no-write Substitute failure; other non-successes stay UNKNOWN.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import NamedTuple
import weakref
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import normalize_id
from . import engine_public_events as public,engine_lifecycle_callbacks as life,mechanics as m
from . import engine_toxic_callbacks as primary,engine_status_callbacks as status

_COMPILED=weakref.WeakKeyDictionary()
Attempt=primary.Attempt  # Existing RAW26 literal guard schema, not Toxic admission.
class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    move_id:object
    substitute_failure:object=None # SAME original source rules; legacy closed
class Context(NamedTuple):
    data:Data
    accounting:object
@dataclass(frozen=True,eq=False)
class Compiled:
    device:Data

def _host():
    if isinstance(j.zeros((),j.int32),jax.core.Tracer):raise ValueError('checked Thunder Wave compiler cannot be traced')
def compile_thunder_wave(*,engine_catalog,metadata,source_rules=None):
    _host();public.check_catalog(engine_catalog,engine_catalog.engine_table,expected_binding=engine_catalog.binding,
        expected_engine_binding=np.asarray(engine_catalog.engine_binding,np.uint32))
    from .engine_private_team import engine_metadata_binding
    if not np.array_equal(engine_metadata_binding(metadata),engine_catalog.engine_binding):raise ValueError('original Thunder Wave metadata required')
    ids=[i for i,name in enumerate(metadata['vocabulary']['moves']) if normalize_id(name)=='thunderwave'];table=engine_catalog.engine_table
    if len(ids)!=1 or ids[0]<=0 or table.effects is None:raise ValueError('canonical Thunder Wave required')
    move=ids[0]
    if (int(table.power[move])!=0 or float(table.accuracy[move])!=100 or bool(table.always_hit[move])
            or int(table.effects.status[move])!=m.PAR or not bool(table.effects.thunderwave[move])
            or engine_catalog.ids[move]<=0 or engine_catalog.targets[move]!=1):
        raise ValueError('canonical Gen3 Thunder Wave100/primary PAR/target required')
    descriptor=None
    if source_rules is not None:
        from . import engine_request_source as request
        origin=request._rules_origin(source_rules)
        if not np.array_equal(source_rules.device.engine_binding,engine_catalog.engine_binding):raise ValueError('same original Thunder Wave rules required')
        if bool(table.effects.bypass_sub[move]) or not bool(table.effects.protectable[move]):raise ValueError('original Thunder Wave Substitute/protection flags required')
        script=r"""
const fs=require('fs'),path=require('path'),crypto=require('crypto'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3'),m=d.moves.get('thunderwave'),s=d.moves.get('substitute').condition;
const h=f=>crypto.createHash('sha256').update(f.toString()).digest('hex');
process.stdout.write(JSON.stringify({sub:h(s.onTryPrimaryHit),getDamage:h(require(path.join(root,'dist/sim/battle-actions.js')).BattleActions.prototype.getDamage),priority:s.onTryPrimaryHitPriority,tryMoveHit:h(require(path.join(root,'dist/data/mods/gen3/scripts.js')).Scripts.actions.tryMoveHit),move:{id:m.id,category:m.category,target:m.target,status:m.status,accuracy:m.accuracy,bypass:!!m.flags.bypasssub},source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));"""
        descriptor=json.loads(subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30).stdout)
        if (descriptor['sub']!='050e1c1fbdb80ea3ab7bc63909eb72433cb45e9c49ff7889e16ec2c207528a24'
                or descriptor['getDamage']!='dfd031900da8c77e5e1a2721858d8d4a7804bc7def1b7a1d486220521a6a7f12'
                or descriptor['priority']!=-1 or descriptor['tryMoveHit']!='af529997d52b6d856301bc8f958ead5963f68abc58d9568c82df24f212b0a97f'
                or descriptor['move']!=dict(id='thunderwave',category='Status',target='normal',status='par',accuracy=100,bypass=False)):
            raise ValueError('original Gen3 Thunder Wave/Substitute source changed')
        if json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original Thunder Wave source tree changed')
    payload=dict(schema='pokezero.thunder-wave-actual-par.v1',substitute_failure=descriptor,move=move,engine=engine_catalog.engine_binding,
        source=engine_catalog.source_binding,moves=engine_catalog.move_binding,producer=engine_catalog.binding)
    words=lambda x:j.asarray(x,j.uint32)
    data=Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),
        words(engine_catalog.engine_binding),words(engine_catalog.source_binding),words(engine_catalog.move_binding),
        words(np.frombuffer(bytes.fromhex(engine_catalog.binding),dtype='>u4').astype(np.uint32)),j.array(move,j.int32),j.array(descriptor is not None,j.bool_))
    result=Compiled(data);_COMPILED[result]=(data,tuple(data),engine_catalog,source_rules);return result

def check_compiled(compiled):
    _host();r=_COMPILED.get(compiled) if type(compiled) is Compiled else None
    if r is None or compiled.device is not r[0] or any(a is not b for a,b in zip(compiled.device,r[1])):raise ValueError('original Thunder Wave compiler required')
    if r[3] is not None:
        from . import engine_request_source as request
        request._rules_origin(r[3])
    catalog=r[2];public.check_catalog(catalog,catalog.engine_table,expected_binding=catalog.binding,
        expected_engine_binding=np.asarray(catalog.engine_binding,np.uint32));return compiled.device

def supported_raw(raw):
    # Successful primary application ONLY. Do not generalize the original
    # Toxic validator, or admit its miss/immunity branches under this source.
    flags=primary.ACCURACY_HIT|primary.FINAL_HIT|primary.LIVE|primary.APPLIED|primary.SECONDARY
    return ((raw[:,0]==primary.TOXIC_ATTEMPT)&(raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,3]==1-raw[:,1])
        &(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,4]>=0)&(raw[:,4]<6)&(raw[:,5]>0)
        &(raw[:,6]==flags)&(raw[:,7]==m.NONE)&(raw[:,8]==m.PAR)&(raw[:,9]==0))


def substitute_failure_source(tape,raw):
    """Explicit checked source capability, never a generic no-write waiver."""
    from . import engine_execution_completeness as ex
    actual=ex._context(tape);b=len(tape.count)
    if actual is None or actual.risks is None:return j.zeros(b,bool)
    tables=actual.risks.tables
    primary._shape(tables.thunder_wave_binding,(8,),j.uint32)
    primary._shape(tables.thunder_wave_substitute_failure,(),j.bool_)
    good=tables.thunder_wave_substitute_failure&j.any(tables.thunder_wave_binding!=0)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        primary._shape(getattr(tape,name),(8,),j.uint32);primary._shape(getattr(tables,name),(8,),j.uint32)
        good &= j.all(getattr(tape,name)==getattr(tables,name))
    move=raw[:,5];safe=j.clip(move,0,len(tables.thunder_wave_candidates)-1)
    return good&primary.substitute_failure_raw(raw)&(move>0)&(move<len(tables.thunder_wave_candidates))&tables.thunder_wave_candidates[safe]


def owned_substitute_failure(tape,index):
    """Recheck the unique current Site19 one-word transaction at consumption."""
    from . import engine_execution_completeness as ex
    actual=ex._context(tape);b=len(tape.count);rows=j.arange(b)
    if actual is None:return j.zeros(b,bool)
    idx=j.broadcast_to(j.asarray(index,j.int32),(b,));at=j.clip(idx,0,tape.words.shape[1]-1)
    raw=tape.words[rows,at];ledger=actual.ledger;claim=ledger.claimed[rows,at]
    slot=j.clip(claim-1,0,ledger.site.shape[1]-1);prior=j.clip(idx-1,0,tape.words.shape[1]-1);decl=tape.words[rows,prior]
    good=substitute_failure_source(tape,raw)&ex._validate(ledger,actual.tables)&ex._tape(ledger,tape,actual.tables)&ledger.opened&(ledger.frame>0)&life._valid(tape.cursor)
    good &= (idx>0)&(idx<tape.count)&(claim>0)&(claim<=ledger.count)&(ledger.site[rows,slot]==int(ex.Site.PRIMARY_PARALYSIS))&(ledger.side[rows,slot]==raw[:,3])&(ledger.stage[rows,slot]==2)&(ledger.start[rows,slot]==idx)&(ledger.end[rows,slot]==idx+1)&(ledger.no_output_kind[rows,slot]==0)
    good &= j.all(ledger.words[rows,at]==raw,axis=1)&j.all(tape.switch_details[rows,at]==0,axis=1)
    good &= (decl[:,0]==public.MOVE)&j.all(decl[:,1:6]==raw[:,1:6],axis=1)&j.all(decl[:,6:]==0,axis=1)
    actor=j.clip(raw[:,1],0,1);target=j.clip(raw[:,3],0,1)
    good &= (raw[:,2]==tape.cursor.physical_active[rows,actor])&(raw[:,4]==tape.cursor.physical_active[rows,target])
    # Consumption follows original upkeep/next-turn advancement. Its
    # exact clock is the current ledger/event snapshot, not the final cursor.
    good &= tape.cursor.initialized&(ledger.phase[rows,slot]==0)&j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
    for pos in (prior,at):good &= (tape.event_turn[rows,pos]==ledger.turn[rows,slot])&(tape.event_phase[rows,pos]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,pos]==ledger.generation[rows,slot],axis=1)
    good &= (ledger.event_turn[rows,at]==tape.event_turn[rows,at])&(ledger.event_phase[rows,at]==tape.event_phase[rows,at])&j.all(ledger.event_generation[rows,at]==tape.event_generation[rows,at],axis=1)
    return good

def _bound(tape,context):
    from . import engine_execution_completeness as ex
    if type(tape) is not life.LifecycleTape or type(context) is not Context or type(context.data) is not Data:raise ValueError('typed Thunder Wave source context required')
    for word in context.data[:5]:primary._shape(word,(8,),j.uint32)
    primary._shape(context.data.move_id,(),j.int32)
    primary._shape(context.data.substitute_failure,(),j.bool_)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        primary._shape(getattr(tape,name),(8,),j.uint32)
    actual=ex._context(tape)
    if actual is None:return j.zeros_like(tape.count,bool)
    good=ex._validate(actual.ledger,context.accounting)&ex._tape(actual.ledger,tape,context.accounting)&actual.ledger.opened&(actual.ledger.refusal==0)&life._valid(tape.cursor)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        good &= j.all(getattr(context.data,name)==getattr(tape,name))
    for before,after in zip(actual.tables,context.accounting):good &= j.all(before==after)
    from . import engine_kernel_frame_risk as risk
    if type(actual.risks) is not risk.Context:return good&False
    rbound,_=risk._bound(actual.risks.carry,tape,actual.risks.tables)
    good &= rbound&j.any(context.data.binding!=0)&j.all(context.data.binding==actual.risks.tables.thunder_wave_binding)
    return (good&(actual.ledger.frame>0)&tape.cursor.initialized&(tape.cursor.phase==0)
        &(tape.turn==tape.cursor.turn)&(context.data.move_id>0))

def span_matches(tape,ticket,side,physical,target,move,attempt,context):
    from . import engine_execution_completeness as ex
    actual=ex._context(tape)
    if actual is None:return j.zeros_like(tape.count,bool)
    ledger=actual.ledger;slot,same=ex._ticket(ledger,ticket);b=len(tape.count);rows=j.arange(b);safe=j.clip(side,0,1)
    raw=primary.words_device(side,physical,target,move,attempt);start=ledger.start[rows,slot]
    first=j.clip(start,0,tape.words.shape[1]-1);at=j.clip(start+1,0,tape.words.shape[1]-1);prefix=j.clip(start-1,0,tape.words.shape[1]-1)
    absent=j.full_like(side,-1);zero=j.zeros_like(side)
    status_word=j.stack((j.full_like(side,status.STATUS_APPLIED),absent,absent,1-side,target,
        attempt.old_status,attempt.new_status,j.full_like(side,status.PRIMARY),zero,zero),axis=1)
    decl=tape.words[rows,prefix]
    failed=context.data.substitute_failure&substitute_failure_source(tape,raw)
    expected_end=start+j.where(failed,1,2)
    good=(_bound(tape,context)&same&(supported_raw(raw)|failed)&(move==context.data.move_id)
        &(attempt.old_counter==0)&(attempt.new_counter==0)&(start>0)&(tape.count==expected_end)
        &(ledger.site[rows,slot]==int(ex.Site.PRIMARY_PARALYSIS))&(ledger.side[rows,slot]==1-side)
        &(physical==tape.cursor.physical_active[rows,safe])&(target==tape.cursor.physical_active[rows,1-safe])
        &(ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)
        &j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
        &j.where(failed,j.all(tape.words[rows,first]==raw,axis=1),j.all(tape.words[rows,first]==status_word,axis=1)&j.all(tape.words[rows,at]==raw,axis=1))
        &(decl[:,0]==public.MOVE)&(decl[:,1]==side)&(decl[:,2]==physical)&(decl[:,3]==1-side)&(decl[:,4]==target)&(decl[:,5]==move)&j.all(decl[:,6:]==0,axis=1))
    for index,required in ((prefix,True),(first,True),(at,~failed)):
        stamp=((tape.event_turn[rows,index]==ledger.turn[rows,slot])&(tape.event_phase[rows,index]==ledger.phase[rows,slot])
            &j.all(tape.event_generation[rows,index]==ledger.generation[rows,slot],axis=1))
        good &= ~j.asarray(required)|stamp
    return good

def record_device(tape,ticket,side,physical,target,move,attempt,context,*,requested):
    from . import engine_execution_completeness as ex
    actual=ex._context(tape)
    if actual is None:return tape
    primary._shape(requested,tape.count.shape,j.bool_)
    raw=primary.words_device(side,physical,target,move,attempt)
    valid=_bound(tape,context)&(supported_raw(raw)|(context.data.substitute_failure&substitute_failure_source(tape,raw)))&(move==context.data.move_id)&(attempt.old_counter==0)&(attempt.new_counter==0)
    tape=public._append(tape,primary.TOXIC_ATTEMPT,side,physical,1-side,target,tuple(raw[:,i] for i in range(5,10)),requested&valid)
    same=span_matches(tape,ticket,side,physical,target,move,attempt,context)
    actual=ex._context(tape);ledger=ex.unhandled(actual.ledger,ticket,active=requested&~same,tables=actual.tables)
    ledger=ex.emitted(ledger,ticket,tape,active=requested&same,tables=actual.tables)
    return ex._put_context(tape,ledger,actual)


def exclude_cure_device(tape,requested,applied,item,item_table):
    """This narrow admission excludes an actual primary-triggered Lum cure.

    Read the original current item/actual apply guard before AfterSetStatus;
    never infer a cure from final status or duplicate its numerical operation.
    The existing reflection guard remains in original after_status_event.
    """
    b=len(tape.count)
    primary._shape(requested,(b,),j.bool_);primary._shape(applied,(b,),j.bool_);primary._shape(item,(b,),j.int32)
    if item_table is None:return tape
    return status.unknown(tape,requested&applied&item_table.lum[item])
