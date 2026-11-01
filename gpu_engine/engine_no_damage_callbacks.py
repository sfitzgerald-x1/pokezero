"""Literal no-damage target callbacks, prospective Site10, never an empty span.

Miss, pure type immunity and the original named TryHit absorbs only. The caller
retains actual operands before TryHit's HP/volatile assignment, then finalizes
against the SAME literal Outcome after the original direct assignment. No HP,
status, ability, hit or effectiveness fact is reconstructed from an afterstate.
"""
from typing import NamedTuple
import hashlib,json,subprocess
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from . import engine_public_events as public,engine_lifecycle_callbacks as life,mechanics as m,absorption

TARGET_NO_DAMAGE,ABSORB_FACTS=33,34
MISS,TYPE_IMMUNE,WATER_HEAL,WATER_FULL,VOLT_HEAL,VOLT_FULL,FLASH_START,FLASH_REPEAT=range(1,9)

class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    candidates:object
    move_types:object
    dynamic_type:object
    dark_type:object
    water:object
    volt:object
    flash:object
    levitate:object
    water_type:object
    electric_type:object
    fire_type:object
    protect_miss:object=None # explicit original accuracy-before-TryHit source

class Before(NamedTuple):
    accuracy_hit:object
    move_type:object
    ability:object
    status:object
    hp:object
    maximum:object
    volatile:object
    sub_hp:object
    actor_hp:object
    explodes:object
    actor_hp_type:object
    actor_hp_power:object

class Pending(NamedTuple):
    context:object
    ticket:object
    actor:object
    physical:object
    target:object
    move:object
    before:Before
    absorbed:object
    requested:object
    caller_move:object=None

class Captured(NamedTuple):
    pending:Pending
    write:object
    actor_hp:object
    volatile:object
    sub_hp:object
    actor_hp_type:object
    actor_hp_power:object

def compile_data(engine_catalog,metadata=None,source_rules=None):
    """Private host compiler from ORIGINAL immutable numerical columns."""
    if isinstance(j.zeros((),j.int32),jax.core.Tracer):raise ValueError('no-damage compiler cannot be traced')
    public.check_catalog(engine_catalog,engine_catalog.engine_table,expected_binding=engine_catalog.binding,
        expected_engine_binding=np.asarray(engine_catalog.engine_binding,np.uint32))
    t=engine_catalog.engine_table
    positive=np.asarray(t.power)>0
    dynamic=np.zeros(len(positive),bool);dark=j.array(17,j.int32)
    if t.variable_moves is not None:
        dynamic=np.asarray(t.variable_moves.hidden_power);dark=t.variable_moves.dark_type
        positive |= dynamic|np.asarray(t.variable_moves.returns)|np.asarray(t.variable_moves.frustration)
    # Reuse only the already represented canonical fixed-LEVEL source row.
    if t.damage_moves is not None:
        seismic=engine_catalog.public_moves.ids.get('seismictoss',0)
        positive |= (np.asarray(engine_catalog.ids)==seismic)&(seismic>0)&np.asarray(t.damage_moves.level_damage)
    candidates=positive&(np.asarray(engine_catalog.ids)>0)&(np.asarray(engine_catalog.targets)==1)
    candidates &= np.arange(len(candidates))!=int(t.struggle)
    if t.damage_moves is not None:candidates &= np.asarray(t.damage_moves.recoil_den)==0
    if t.self_destruct is not None:candidates &= ~np.asarray(t.self_destruct.move)
    a=t.absorption
    if a is None or t.entry is None:
        candidates[:]=False
        columns=tuple(j.zeros(1,bool)for _ in range(4));type_ids=(j.array(0,j.int32),)*3
    else:
        columns=(a.water_absorb,a.volt_absorb,a.flash_fire,t.entry.levitate)
        type_ids=(a.water_type,a.electric_type,a.fire_type)
    if any(v.shape!=columns[0].shape or v.dtype!=j.dtype(bool)for v in columns):raise ValueError('original ability namespace required')
    descriptor=None
    if metadata is not None or source_rules is not None:
        if metadata is None or source_rules is None:raise ValueError('original no-damage metadata AND rules required')
        from . import engine_request_source as request
        from .engine_private_team import engine_metadata_binding
        origin=request._rules_origin(source_rules)
        if not np.array_equal(engine_metadata_binding(metadata),engine_catalog.engine_binding) or not np.array_equal(source_rules.device.engine_binding,engine_catalog.engine_binding):raise ValueError('same original no-damage rules/catalog required')
        script=r"""
const fs=require('fs'),path=require('path'),crypto=require('crypto'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const checked=require(process.argv[2]).build(root,meta),f=require(path.join(root,'dist/data/mods/gen3/scripts.js')).Scripts.actions.tryMoveHit;
process.stdout.write(JSON.stringify({tryMoveHit:crypto.createHash('sha256').update(f.toString()).digest('hex'),source_tree:checked.source_tree,compiled_tree:checked.compiled_tree}));"""
        descriptor=json.loads(subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30).stdout)
        if descriptor['tryMoveHit']!='af529997d52b6d856301bc8f958ead5963f68abc58d9568c82df24f212b0a97f':raise ValueError('original Gen3 accuracy-before-TryHit source required')
        if json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original no-damage source changed')
    payload=dict(schema='pokezero.literal-no-damage-target.v2',protect_miss_source=descriptor,producer=engine_catalog.binding,
        candidates=candidates.tolist(),move_types=np.asarray(t.type).tolist(),dynamic_type=dynamic.tolist(),dark_type=int(dark),abilities=[np.asarray(v).tolist()for v in columns],
        source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    words=lambda v:j.asarray(v,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),
        words(engine_catalog.engine_binding),words(engine_catalog.source_binding),words(engine_catalog.move_binding),
        words(np.frombuffer(bytes.fromhex(engine_catalog.binding),dtype='>u4').astype(np.uint32)),j.array(candidates),t.type,j.array(dynamic),dark,
        *columns,*type_ids,j.array(descriptor is not None,j.bool_))

def _shape(data):
    if type(data)is not Data:raise ValueError('original no-damage Data required')
    for v in data[:5]:life._typed(v,(8,),j.uint32)
    life._typed(data.candidates,(len(data.candidates),),j.bool_)
    life._typed(data.move_types,data.candidates.shape,j.int32)
    life._typed(data.dynamic_type,data.candidates.shape,j.bool_)
    life._typed(data.dark_type,(),j.int32);life._typed(data.protect_miss,(),j.bool_)
    for v in (data.water,data.volt,data.flash,data.levitate):life._typed(v,(len(data.water),),j.bool_)
    for v in (data.water_type,data.electric_type,data.fire_type):life._typed(v,(),j.int32)

def prepare(tape,ticket,actor,physical,target,move,before,context,*,requested,caller_move=None):
    if type(before)is not Before:raise ValueError('literal pre-TryHit operands required')
    return Pending(context,ticket,actor,physical,target,move,before,
        absorption.AbsorptionResult(j.zeros_like(requested),before.hp,before.volatile),requested,caller_move)

def observed_absorption(pending,result):
    if type(result)is not absorption.AbsorptionResult:raise ValueError('original literal TryHit result required')
    return pending._replace(absorbed=result)

def _code(p,w):
    from . import engine_direct_hit_controls as direct
    data=p.context.data.no_damage;_shape(data);b=len(p.actor);before=p.before;a=p.absorbed
    w=direct._typed_write(w,b)
    for value in (p.actor,p.physical,p.target,p.move,before.move_type,before.ability,before.status,before.hp,before.maximum,before.sub_hp,before.actor_hp,before.actor_hp_type,before.actor_hp_power):life._typed(value,(b,),j.int32)
    for value in (before.volatile,a.volatile):life._typed(value,(b,),j.uint32)
    for value in (p.requested,before.accuracy_hit,before.explodes,a.blocked):life._typed(value,(b,),j.bool_)
    life._typed(a.hp,(b,),j.int32)
    selected_type=data.move_types[j.clip(p.move,0,len(data.candidates)-1)]
    dynamic=data.dynamic_type[j.clip(p.move,0,len(data.candidates)-1)]
    # Join the literal resolved type to its actual current member selector;
    # neither type effectiveness nor damage is recomputed here.
    type_join=j.where(dynamic,j.where(before.actor_hp_type>0,before.move_type==before.actor_hp_type,before.move_type==data.dark_type),before.move_type==selected_type)
    fixed=direct.fixed_level_move_device(p.context.data,p.move)
    positive_ratio=direct._positive_ratio(w.numerator,w.denominator)
    finite=positive_ratio|((w.numerator==0)&((w.denominator==1)|(w.denominator==2)))
    valid=(finite&p.requested&(p.move>0)&(p.move<len(data.candidates))&data.candidates[j.clip(p.move,0,len(data.candidates)-1)]
        &type_join&(before.actor_hp_type>=0)&(before.actor_hp_type<18)&(before.actor_hp_power>=0)&(before.actor_hp_power<=70)
        &(before.ability>=0)&(before.ability<len(data.water))&(before.move_type>0)&(before.move_type<18)
        &(before.hp>0)&(before.hp<=before.maximum)&(before.actor_hp>0)&(before.status>=0)&(before.status<=6)
        &(before.sub_hp==0)&~before.explodes&((w.power>0)|((w.power==0)&fixed))&~w.substitute&~w.enduring
        &(w.old_hp==a.hp)&(w.new_hp==a.hp)&(w.maximum==before.maximum))
    abi=j.clip(before.ability,0,len(data.water)-1)
    same=(a.hp==before.hp)&(a.volatile==before.volatile)
    miss=~before.accuracy_hit&~w.hit&~w.immune&positive_ratio&~a.blocked&same
    pure=before.accuracy_hit&w.hit&w.immune&(w.numerator==0)&((w.denominator==1)|(w.denominator==2))&~a.blocked&~data.levitate[abi]&same
    absorb=before.accuracy_hit&~w.hit&~w.immune&positive_ratio&a.blocked
    healed=absorb&(a.hp>before.hp)&(a.hp<=before.maximum)&(a.volatile==before.volatile)
    full=absorb&(before.hp==before.maximum)&same
    water=data.water[abi]&(before.move_type==data.water_type)
    volt=data.volt[abi]&(before.move_type==data.electric_type)
    flash=absorb&data.flash[abi]&(before.move_type==data.fire_type)&(before.status!=m.FRZ)&(a.hp==before.hp)
    flash &= a.volatile==(before.volatile|j.uint32(absorption.FLASH_FIRE))
    code=j.where(miss,MISS,j.where(pure,TYPE_IMMUNE,j.where(water&healed,WATER_HEAL,j.where(water&full,WATER_FULL,
        j.where(volt&healed,VOLT_HEAL,j.where(volt&full,VOLT_FULL,j.where(flash&((before.volatile&absorption.FLASH_FIRE)==0),FLASH_START,
        j.where(flash,FLASH_REPEAT,0))))))))
    protected_miss=j.zeros(b,bool)
    if p.caller_move is not None and p.context.data.protect is not None:
        from . import engine_protect_callbacks as protect
        life._typed(p.caller_move,(b,),j.int32);protect.shape(p.context.data.protect)
        source_move=p.context.data.protect.block_candidates[j.clip(p.move,0,len(data.candidates)-1)]
        protected_miss=data.protect_miss&miss&w.protected&(p.caller_move==p.move)&source_move
    return j.where(valid&(~w.protected|protected_miss),code,0)

def scheduled(pending,write):
    """Typed prospective branch; final literal Outcome still MUST validate."""
    return _code(pending,write)>0

def words(p,w):
    before=p.before;code=_code(p,w);z=j.zeros_like(p.actor)
    header=j.stack((z+TARGET_NO_DAMAGE,p.actor,p.physical,1-p.actor,p.target,p.move,code,before.hp,p.absorbed.hp,before.maximum),axis=1)
    facts=j.stack((z+ABSORB_FACTS,p.actor,p.physical,1-p.actor,p.target,p.move,before.ability,before.status,
        jax.lax.bitcast_convert_type(before.volatile,j.int32),jax.lax.bitcast_convert_type(p.absorbed.volatile,j.int32)),axis=1)
    return header,facts

def valid_raw_pair(header,facts,data):
    _shape(data);move=header[:,5];code=header[:,6];abi=facts[:,6];safe=j.clip(abi,0,len(data.water)-1)
    before=jax.lax.bitcast_convert_type(facts[:,8],j.uint32);after=jax.lax.bitcast_convert_type(facts[:,9],j.uint32)
    typ=data.move_types[j.clip(move,0,len(data.candidates)-1)]
    dynamic=data.dynamic_type[j.clip(move,0,len(data.candidates)-1)]
    same=(header[:,7]==header[:,8])&(before==after)
    return ((header[:,0]==TARGET_NO_DAMAGE)&(facts[:,0]==ABSORB_FACTS)&j.all(header[:,1:6]==facts[:,1:6],axis=1)
        &(header[:,1]>=0)&(header[:,1]<2)&(header[:,3]==1-header[:,1])&(header[:,2]>=0)&(header[:,2]<6)&(header[:,4]>=0)&(header[:,4]<6)
        &(move>0)&(move<len(data.candidates))&data.candidates[j.clip(move,0,len(data.candidates)-1)]
        &(abi>=0)&(abi<len(data.water))&(facts[:,7]>=0)&(facts[:,7]<=6)&(header[:,7]>0)&(header[:,7]<=header[:,9])&(header[:,8]>0)&(header[:,8]<=header[:,9])
        &(((code==MISS)|(code==TYPE_IMMUNE))&same
            |((code==WATER_HEAL)&data.water[safe]&((typ==data.water_type)|dynamic)|(code==VOLT_HEAL)&data.volt[safe]&((typ==data.electric_type)|dynamic))&(header[:,8]>header[:,7])&(before==after)
            |((code==WATER_FULL)&data.water[safe]&((typ==data.water_type)|dynamic)|(code==VOLT_FULL)&data.volt[safe]&((typ==data.electric_type)|dynamic))&same&(header[:,7]==header[:,9])
            |((code==FLASH_START)|(code==FLASH_REPEAT))&data.flash[safe]&((typ==data.fire_type)|dynamic)&(facts[:,7]!=m.FRZ)&(header[:,7]==header[:,8])
              &(after==(before|j.uint32(absorption.FLASH_FIRE)))&(((before&absorption.FLASH_FIRE)==0)==(code==FLASH_START))))

def _bound(tape,p):
    from . import engine_direct_hit_controls as direct,engine_execution_completeness as ex
    good=direct._registered_bound(tape,p.context);actual=ex._context(tape)
    if actual is None:return good&False
    data=p.context.data.no_damage
    if getattr(actual,'risks',None)is None:return good&False
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):good &= j.all(getattr(data,name)==getattr(tape,name))
    life._typed(actual.risks.tables.direct_hit_binding,(8,),j.uint32)
    life._typed(actual.risks.tables.no_damage_binding,(8,),j.uint32)
    good &= j.all(data.binding==actual.risks.tables.no_damage_binding)&j.any(data.binding!=0)
    good &= (p.context.data.move_count==len(data.candidates))&tape.cursor.initialized&(tape.cursor.phase==0)
    good &= j.all(p.context.data.binding==actual.risks.tables.direct_hit_binding)
    return good

def matches(tape,p,w,outcome,actor_hp,volatile,sub_hp,actor_hp_type,actor_hp_power,*,stage):
    from . import engine_execution_completeness as ex
    actual=ex._context(tape)
    if actual is None:return j.zeros_like(tape.count,bool)
    ledger=actual.ledger;slot,same=ex._ticket(ledger,p.ticket);rows=j.arange(len(p.actor));start=ledger.start[rows,slot]
    header,facts=words(p,w);code=header[:,6];source=_bound(tape,p)&same&(code>0)
    source &= (outcome.power==w.power)&(outcome.critical==w.critical)&(outcome.hit==w.hit)&(outcome.immune==w.immune)
    source &= (outcome.numerator==w.numerator)&(outcome.denominator==w.denominator)&(outcome.absorbed==p.absorbed.blocked)
    source &= (outcome.substitute==w.substitute)&(outcome.protected==w.protected)&(outcome.enduring==w.enduring)
    source &= (actor_hp_type==p.before.actor_hp_type)&(actor_hp_power==p.before.actor_hp_power)
    source &= (outcome.recoil==0)&~outcome.explodes&(actor_hp==p.before.actor_hp)&(volatile==p.absorbed.volatile)&(sub_hp==p.before.sub_hp)
    safe=j.clip(p.actor,0,1)
    source &= (ledger.site[rows,slot]==int(ex.Site.DIRECT_HIT))&(ledger.side[rows,slot]==1-p.actor)&(ledger.stage[rows,slot]==stage)
    source &= life._valid(tape.cursor)&(p.physical==tape.cursor.physical_active[rows,safe])&(p.target==tape.cursor.physical_active[rows,1-safe])
    source &= (tape.cursor.turn==ledger.turn[rows,slot])&(tape.cursor.phase==ledger.phase[rows,slot])&j.all(tape.cursor.generation==ledger.generation[rows,slot],axis=1)
    prefix=j.clip(start-1,0,tape.words.shape[1]-1);decl=tape.words[rows,prefix]
    source &= (start>0)&(decl[:,0]==public.MOVE)&j.all(decl[:,1:6]==header[:,1:6],axis=1)&j.all(decl[:,6:]==0,axis=1)
    length=0 if stage==1 else 2;source &= tape.count==start+length
    for at in (prefix,):
        source &= (tape.event_turn[rows,at]==ledger.turn[rows,slot])&(tape.event_phase[rows,at]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,at]==ledger.generation[rows,slot],axis=1)
    if stage==2:
        for offset,expected in ((0,header),(1,facts)):
            at=j.clip(start+offset,0,tape.words.shape[1]-1)
            source &= j.all(tape.words[rows,at]==expected,axis=1)&j.all(ledger.words[rows,at]==expected,axis=1)&(ledger.claimed[rows,at]==slot+1)
            source &= (tape.event_turn[rows,at]==ledger.turn[rows,slot])&(tape.event_phase[rows,at]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,at]==ledger.generation[rows,slot],axis=1)
    return source

def finish(tape,p,w,outcome,actor_hp,volatile,sub_hp,actor_hp_type,actor_hp_power):
    from . import engine_execution_completeness as ex
    requested=p.requested&scheduled(p,w);good=matches(tape,p,w,outcome,actor_hp,volatile,sub_hp,actor_hp_type,actor_hp_power,stage=1)
    room=(tape.count>=0)&(tape.count<=tape.words.shape[1]-2);use=requested&good&room
    header,facts=words(p,w)
    for kind,raw in ((TARGET_NO_DAMAGE,header),(ABSORB_FACTS,facts)):tape=public._append(tape,kind,p.actor,p.physical,1-p.actor,p.target,tuple(raw[:,i]for i in range(5,10)),use)
    actual=ex._context(tape)
    ledger=ex.unhandled(actual.ledger,p.ticket,active=requested&~(good&room),tables=actual.tables)
    ledger=ex.emitted(ledger,p.ticket,tape,active=use,tables=actual.tables)
    tape=ex._put_context(tape,ledger,actual)
    return tape._replace(refusal=tape.refusal|j.where(requested&~(good&room),256,0)|j.where(requested&good&~room,public.OVERFLOW,0))

def captured_matches(tape,captured,outcome):
    if type(captured)is not Captured:return j.zeros_like(tape.count,bool)
    return matches(tape,captured.pending,captured.write,outcome,captured.actor_hp,captured.volatile,captured.sub_hp,captured.actor_hp_type,captured.actor_hp_power,stage=2)

def owned_pair(tape,index,data):
    """Completed captured Site10 snapshot at FULL time, not after-frame clocks."""
    from . import engine_execution_completeness as ex
    b=len(tape.count);rows=j.arange(b);raw=tape.words[:,index]
    start=j.where(raw[:,0]==ABSORB_FACTS,index-1,index)
    first=j.clip(start,0,tape.words.shape[1]-1);second=j.clip(start+1,0,tape.words.shape[1]-1)
    header=tape.words[rows,first];facts=tape.words[rows,second];actual=ex._context(tape)
    if actual is None:return j.zeros(b,bool),header,facts
    ledger=actual.ledger;claim=ledger.claimed[rows,first];slot=j.clip(claim-1,0,ledger.site.shape[1]-1)
    good=ex._validate(ledger,actual.tables)&ex._tape(ledger,tape,actual.tables)&valid_raw_pair(header,facts,data)
    good &= (start>0)&(start+1<tape.count)&(claim>0)&(claim<=ledger.count)&(ledger.claimed[rows,second]==claim)
    good &= (ledger.site[rows,slot]==int(ex.Site.DIRECT_HIT))&(ledger.stage[rows,slot]==2)&(ledger.start[rows,slot]==start)&(ledger.end[rows,slot]==start+2)
    prefix=j.clip(start-1,0,tape.words.shape[1]-1);decl=tape.words[rows,prefix]
    good &= (decl[:,0]==public.MOVE)&j.all(decl[:,1:6]==header[:,1:6],axis=1)&j.all(decl[:,6:]==0,axis=1)
    for at in (prefix,first,second):
        good &= (tape.event_turn[rows,at]==ledger.turn[rows,slot])&(tape.event_phase[rows,at]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,at]==ledger.generation[rows,slot],axis=1)
    for at in (first,second):
        good &= j.all(ledger.words[rows,at]==tape.words[rows,at],axis=1)
        good &= (ledger.event_turn[rows,at]==tape.event_turn[rows,at])&(ledger.event_phase[rows,at]==tape.event_phase[rows,at])&j.all(ledger.event_generation[rows,at]==tape.event_generation[rows,at],axis=1)
    return good,header,facts
