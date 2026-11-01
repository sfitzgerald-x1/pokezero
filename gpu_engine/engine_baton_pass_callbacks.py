"""PRIVATE two-boundary ordinary boost/Substitute Baton transaction; no physics.

RAW54 START=[54,side,old,move,0,turn,frame,generation,publicSlot,secondary].
RAW54 INSTALL=[54,side,old,new,1,turn,frame,startGeneration,publicSlot,0]
followed by the genuine RAW6 installation with its truthful Baton flag1.
Missing public history, private request or unported transfer remains UNKNOWN.
"""
import hashlib,json,subprocess
from pathlib import Path
from typing import NamedTuple
import jax
import jax.numpy as j
import numpy as np
from . import engine_lifecycle_callbacks as life,engine_public_events as public

BATON_PASS=54
TOXIC_SWITCH_IN_SHA256='36fcc82af87dfb5bbcdc39bdfcf2e763d57721d27df52e5fcce3afc462aff645'
COPY_VOLATILE_SHA256='c3499e9e26b9f935bc8eacc3c6620d03f00263e483d5dd24abf99b56bed1a6db'
CLEAR_VOLATILE_SHA256='bad93b384074efda34ac198ef3acabbed50f249f6464d9a030a4181a10839885'
PAR_BEFORE_MOVE_SHA256='713407730186869cd1e77e250eec0ff6ceb40b018e884725ebcbff59cbbaf66f'
INSOMNIA_UPDATE_SHA256='259090f315b83537cc5d7091bc08a37a051007304d7392fb6e8ac90a5d4fca66'
NATURAL_CURE_SWITCH_OUT_SHA256='bbd17c399abcaecf20ec77369b6228b574b1a451a448b202daccc7e9210c3cc4'
TOXIC_STATUS=4
PARALYSIS_STATUS=1
BURN_STATUS,POISON_STATUS=2,3
START,INSTALL=0,1
PRIVATE_FIELDS=('hp','max_hp','status','species','level','gender','stats','moves','pp','item','ability','base_ability','types','base_types','hidden_power_type','hidden_power_power','happiness','status_counter','substitute_hp')
Private=NamedTuple('Private',[(n,object)for n in PRIVATE_FIELDS])
SUSPENSION_FIELDS=('turn','pending_residual','baton_pass','forced_switch','queued_actor','queued_actions','queued_stall','residual_draws','phazing_draws','secondary_draws','sleep_talk_draws','replacement_count','keys')
Suspension=NamedTuple('Suspension',[(n,object)for n in SUSPENSION_FIELDS])
class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    move_id:object
    quiet_abilities:object
    quiet_items:object
    outgoing_dormant_abilities:object=None
    outgoing_insomnia_status:object=None
class History(NamedTuple):
    stream_id:object
    frame:object
    turn:object
    generation:object
    active:object
    side:object
    public_slot:object
    ready:object
    boosts:object
    private:object
    order:object
    suspended:object
class PublicProof(NamedTuple):
    boosts:object
    known:object
    pending:object
    substitute:object
class Record(NamedTuple):
    known:object
    finalized:object
    code:object
    start:object
    slot:object
    serial:object
    frame:object
    side:object
    outgoing:object
    incoming:object
    secondary:object
    history:object
    before:object
    boost_before:object
    boost_after:object
    after:object
class Transaction(NamedTuple):
    history:object
    public:object
    request_ok:object
    recipient:object
    order:object
    record:object
class Pending(NamedTuple):
    ticket:object
    requested:object
    valid:object
    side:object
    physical:object
    incoming:object
    history:object
    before:object
    boosts:object

def compile_data(catalog,metadata,rules):
    from . import engine_request_source as request
    from .engine_private_team import engine_metadata_binding
    origin=request._rules_origin(rules)
    if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding)or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):raise ValueError('same original Baton metadata/rules required')
    script=r"""
const fs=require('fs'),path=require('path'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const checked=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3'),m=d.moves.get('batonpass'),P=require(path.join(root,'dist/sim/pokemon.js')).Pokemon;
const hooks=['onStart','onEnd','onBeforeSwitchOut','onSwitchOut','onSwitchIn','onAnySwitchIn','onAllySwitchIn','onUpdate'];
const quiet=(e)=>e.exists&&e.gen<=3&&!hooks.some(n=>typeof e[n]!=='undefined');
const dormantHook={insomnia:'onUpdate',naturalcure:'onSwitchOut'};
const dormant=(e)=>e.exists&&e.gen<=3&&Object.hasOwn(dormantHook,e.id)&&typeof e[dormantHook[e.id]]==='function'&&!hooks.filter(n=>n!==dormantHook[e.id]).some(n=>typeof e[n]!=='undefined');
const passiveStatusHooks=Object.fromEntries(['brn','psn'].map(id=>[id,Object.fromEntries(['onBeforeSwitchOut','onSwitchOut','onSwitchIn','onCopy'].map(n=>[n,typeof d.conditions.get(id)[n]]))]));
const dormantSources=Object.fromEntries(Object.entries(dormantHook).map(([id,hook])=>{const e=d.abilities.get(id);return[id,{hook,body:''+e[hook],otherHooks:hooks.filter(n=>n!==hook).map(n=>typeof e[n])}]}));
process.stdout.write(JSON.stringify({source_tree:checked.source_tree,compiled_tree:checked.compiled_tree,move:{id:m.id,gen:m.gen,category:m.category,target:m.target,power:m.basePower,accuracy:m.accuracy,priority:m.priority,selfSwitch:m.selfSwitch,onHit:(''+m.onHit),selfHit:(''+m.self.onHit)},toxicSwitchIn:''+d.conditions.get('tox').onSwitchIn,copyVolatile:''+P.prototype.copyVolatileFrom,clearVolatile:''+P.prototype.clearVolatile,paralysis:{beforeMove:''+d.conditions.get('par').onBeforeMove,beforeSwitchOut:typeof d.conditions.get('par').onBeforeSwitchOut,switchOut:typeof d.conditions.get('par').onSwitchOut,switchIn:typeof d.conditions.get('par').onSwitchIn},substitute:{id:d.conditions.get('substitute').id,noCopy:d.conditions.get('substitute').noCopy,onCopy:typeof d.conditions.get('substitute').onCopy},passiveStatusHooks,outgoingInsomniaStatus:meta.vocabulary.abilities.map(n=>{const e=d.abilities.get(n);return e.id==='insomnia'&&dormant(e)}),dormantSources,outgoingDormant:meta.vocabulary.abilities.map(n=>dormant(d.abilities.get(n))),abilities:meta.vocabulary.abilities.map(n=>quiet(d.abilities.get(n))),items:meta.vocabulary.items.map((n,i)=>!i||quiet(d.items.get(n)))}));"""
    value=json.loads(subprocess.check_output([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata).encode(),timeout=30))
    if json.dumps(value['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(value['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original Baton source drift')
    if hashlib.sha256(value['toxicSwitchIn'].encode()).hexdigest()!=TOXIC_SWITCH_IN_SHA256:raise ValueError('canonical original toxic switch reset required')
    if hashlib.sha256(value['copyVolatile'].encode()).hexdigest()!=COPY_VOLATILE_SHA256:raise ValueError('canonical original Baton copy required')
    if hashlib.sha256(value['clearVolatile'].encode()).hexdigest()!=CLEAR_VOLATILE_SHA256:raise ValueError('canonical original outgoing major-status preservation required')
    par=value['paralysis']
    if hashlib.sha256(par['beforeMove'].encode()).hexdigest()!=PAR_BEFORE_MOVE_SHA256 or any(par[n]!='undefined'for n in ('beforeSwitchOut','switchOut','switchIn')):raise ValueError('canonical original paralysis before-move and quiet departure required')
    if value['substitute']!=dict(id='substitute',noCopy=False,onCopy='undefined'):raise ValueError('canonical original Substitute transfer required')
    for name,hook,digest in [('insomnia','onUpdate',INSOMNIA_UPDATE_SHA256),('naturalcure','onSwitchOut',NATURAL_CURE_SWITCH_OUT_SHA256)]:
        actual=value['dormantSources'][name]
        if actual['hook']!=hook or hashlib.sha256(actual['body'].encode()).hexdigest()!=digest or actual['otherHooks']!=['undefined']*7:raise ValueError('canonical original dormant outgoing ability required')
    if value['passiveStatusHooks']!={n:{hook:'undefined'for hook in('onBeforeSwitchOut','onSwitchOut','onSwitchIn','onCopy')}for n in('brn','psn')}:raise ValueError('canonical original burn/poison quiet departure required')
    row=value['move'];names=metadata['vocabulary']['moves'];ids=[i for i,n in enumerate(names)if n=='batonpass']
    move=ids[0]if len(ids)==1 else 0;t=catalog.engine_table
    if move:
        if not(row['id']=='batonpass'and row['gen']==2 and row['category']=='Status'and row['target']=='self'and row['power']==0 and row['accuracy']is True and row['priority']==0 and row['selfSwitch']=='copyvolatile'and t.baton is not None and bool(t.baton.move[move])and int(t.power[move])==0 and bool(t.always_hit[move])and int(catalog.ids[move])>0 and catalog.targets[move]==0):raise ValueError('canonical original Baton columns required')
    payload=dict(schema='pokezero.baton.boost-substitute.toxic-paralysis-insomnia-brn-psn-source.v6',source=value,engine=catalog.engine_binding,moves=catalog.move_binding,producer=catalog.binding,move=move)
    words=lambda v:j.asarray(v,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(move,j.int32),j.asarray(value['abilities']),j.asarray(value['items']),j.asarray(value['outgoingDormant']),j.asarray([BURN_STATUS,POISON_STATUS,len(value['abilities'])]+[int(x)for x in value['outgoingInsomniaStatus']],j.int32))

def shape(d):
    if type(d)is not Data:raise ValueError('typed original Baton capability required')
    for v in d[:5]:life._typed(v,(8,),j.uint32)
    life._typed(d.move_id,(),j.int32)
    for v in d[6:8]:life._typed(v,(len(v),),j.bool_)
    if d.outgoing_dormant_abilities is not None:life._typed(d.outgoing_dormant_abilities,d.quiet_abilities.shape,j.bool_)
    if d.outgoing_insomnia_status is not None:life._typed(d.outgoing_insomnia_status,(3+len(d.quiet_abilities),),j.int32)
def snapshot(s):return Private(*(getattr(s,n)for n in PRIVATE_FIELDS))
def suspend(s):return Suspension(*(getattr(s,n)for n in SUSPENSION_FIELDS))
def equal(a,b):
    out=j.ones(len(a[0]),bool)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):out &= j.all(x==y,axis=tuple(range(1,x.ndim)))if x.ndim>1 else x==y
    return out
def rows(use,a,b):return jax.tree.map(lambda x,y:j.where(use.reshape((len(use),)+(1,)*(x.ndim-1)),y,x),a,b)
def empty_history(state,cursor,order):
    b=len(cursor.stream_id);return History(cursor.stream_id,j.full(b,-1,j.int32),cursor.turn,cursor.generation,state.active,j.full(b,-1,j.int32),j.full(b,-1,j.int32),j.zeros(b,bool),state.boosts,snapshot(state),order,suspend(state))
def history_shape(h,b):
    if type(h)is not History or type(h.private)is not Private or type(h.suspended)is not Suspension:raise ValueError('typed same-source Baton history required')
    for n in ('stream_id','frame','turn','side','public_slot'):life._typed(getattr(h,n),(b,),j.int32)
    life._typed(h.ready,(b,),j.bool_);life._typed(h.generation,(b,2),j.int32);life._typed(h.active,(b,2),j.int32);life._typed(h.boosts,(b,2,6,7),j.int32);life._typed(h.order,(b,2,6),j.int32)
    for name,v in zip(PRIVATE_FIELDS,h.private,strict=True):
        tail=(6,)if name=='stats'else(4,)if name in ('moves','pp')else(2,)if name in ('types','base_types')else()
        life._typed(v,(b,2,6)+tail,j.int32)
    suspension_shape(h.suspended,b)
def suspension_shape(s,b):
    if type(s)is not Suspension:raise ValueError('typed same-source Baton suspension required')
    for name,v in zip(SUSPENSION_FIELDS,s,strict=True):
        tail=(2,12)if name=='residual_draws'else(2,)if name in ('baton_pass','forced_switch','queued_actions','queued_stall','phazing_draws','secondary_draws','sleep_talk_draws','keys')else()
        dtype=j.bool_ if name in ('pending_residual','baton_pass','forced_switch')else j.uint32 if name=='keys'else j.int32
        life._typed(v,(b,)+tail,dtype)

def empty_record(h):
    b=len(h.side);z=j.zeros(b,j.int32);return Record(j.zeros(b,bool),j.zeros(b,bool),z,z,z,z,z,h.side,h.side,h.side,j.zeros(b,bool),h,h.private,h.boosts,h.boosts,h.private)
def _bound(tape,data):
    from . import engine_execution_completeness as ex
    shape(data);c=ex._context(tape)
    if c is None or c.risks is None:return j.zeros(len(tape.count),bool)
    good=ex._validate(c.ledger,c.tables)&ex._tape(c.ledger,tape,c.tables)&j.any(data.binding!=0)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);good &= j.all(getattr(data,name)==getattr(tape,name))
    t=c.risks.tables
    for n,v in [('binding',data.binding),('move',data.move_id),('abilities',data.quiet_abilities),('items',data.quiet_items)]:
        actual=getattr(t,'baton_'+n);life._typed(actual,v.shape,v.dtype);good &= j.all(actual==v)
    if data.outgoing_dormant_abilities is not None:
        actual=t.baton_outgoing_dormant_abilities;life._typed(actual,data.outgoing_dormant_abilities.shape,j.bool_);good &= j.all(actual==data.outgoing_dormant_abilities)
    if data.outgoing_insomnia_status is not None:
        actual=t.baton_outgoing_insomnia_status;life._typed(actual,data.outgoing_insomnia_status.shape,j.int32);good &= j.all(actual==data.outgoing_insomnia_status)
    return good

def _data(tape):
    t=tape.execution.risks.tables
    cap=getattr(t,'baton_outgoing_insomnia_status',None)
    if cap is not None and cap.shape==(0,):cap=None
    return Data(t.baton_binding,t.engine_binding,t.source_binding,t.move_binding,t.producer_binding,t.baton_move,t.baton_abilities,t.baton_items,getattr(t,'baton_outgoing_dormant_abilities',None),cap)
def public_substitute(pub):
    """Literal original public volatile namespace; every other bit stays closed."""
    from . import public_active
    v=pub.context.active.volatiles;b=len(pub.cursor.stream_id)
    life._typed(v,(b,2,len(public_active.NAMES)),j.bool_)
    sid=public_active.NAMES.index('substitute')
    return v[:,:,sid],~j.any(v&(j.arange(v.shape[-1])!=sid),axis=2)

def public_proof(state,cursor,publics,healthy,data):
    """Actual BOTH before views; zero public facts are never inferred from engine."""
    b=len(cursor.stream_id);known=j.broadcast_to(healthy[:,None],(b,2));boosts=[];pending=[];substitute=[]
    for owner,p in enumerate(publics):
        coherent=(p.cursor.stream_id==cursor.stream_id)&(p.cursor.turn==cursor.turn)&(p.cursor.phase==cursor.phase)&j.all(p.mons.active==cursor.public_slot,axis=1)&j.all(p.context.active.encounter==cursor.encounter,axis=1)
        sub,quiet=public_substitute(p)
        known &= coherent[:,None]&quiet
        boosts.append(p.mons.boosts);pending.append(p.context.tendency.baton);substitute.append(sub)
    known &= j.all(boosts[0]==boosts[1],axis=2)&(pending[0]==pending[1])&(substitute[0]==substitute[1])
    return PublicProof(boosts[0],known,pending[0],substitute[0])
def prepare(tape,state,h,proof,requests,orders,actions,data):
    """The SAME already-produced request/party permutation, no replacement guess."""
    from . import private_requests as req
    b=len(tape.count);r=j.arange(b);side=j.clip(h.side,0,1);target=actions[r,side]-4;order=j.stack(tuple(x.request_to_physical for x in orders),axis=1)
    history_shape(h,b);suspension_shape(suspend(state),b)
    life._typed(proof.boosts,(b,2,7),j.int32);life._typed(proof.known,(b,2),j.bool_);life._typed(proof.pending,(b,2),j.bool_);life._typed(proof.substitute,(b,2),j.bool_)
    good=_bound(tape,data)&h.ready&equal(h.suspended,suspend(state))&(h.side>=0)&(h.frame+1==tape.execution.ledger.frame)&(h.stream_id==tape.stream_id)&(h.turn==tape.cursor.turn)&j.all(h.generation==tape.cursor.generation,axis=1)&j.all(h.active==state.active,axis=1)&equal(h.private,snapshot(state))&j.all(h.boosts==state.boosts,axis=(1,2,3))&j.all(order==h.order,axis=(1,2))
    for owner,q in enumerate(requests):
        actual=q.actions;packet=actual.facts.request;own=h.side==owner
        good &= j.all(actual.boundary.engine_binding==tape.engine_binding,axis=1)&(orders[owner].generation==h.generation[:,owner])&j.all(orders[owner].engine_binding==tape.engine_binding,axis=1)
        good &= actual.admitted&(actual.boundary.stream_id==tape.stream_id)&(actual.boundary.owner_side==owner)&(actual.boundary.generation==orders[owner].generation)&(actual.boundary.active_physical==state.active[:,owner])
        good &= j.where(own,(packet.kind==int(req.RequestKind.FORCE_SWITCH))&packet.force_switch&~packet.wait,(packet.kind==int(req.RequestKind.WAIT))&packet.wait&~packet.force_switch)
        safe=j.clip(actions[:,owner],0,actual.engine_legal.shape[1]-1);good &= ~own|actual.engine_legal[r,safe]
    good &= (target>=0)&(target<6)&(target!=h.active[r,side])&proof.known[r,side]&proof.pending[r,side]&j.all(proof.boosts[r,side]==h.boosts[r,side,h.active[r,side]],axis=1)
    good &= proof.substitute[r,side]==(h.private.substitute_hp[r,side,h.active[r,side]]>0)
    tr=Transaction(h,proof,good,j.clip(target,0,5),order,empty_record(h))
    return tape._replace(execution=tape.execution._replace(baton_transaction=tr))

def _insomnia_status(ability,status,counter,data):
    """Canonical Insomnia's onUpdate is dormant only under these bound facts."""
    cap=getattr(data,'outgoing_insomnia_status',None)
    if cap is None:return j.zeros_like(status,bool)
    life._typed(cap,(3+len(data.quiet_abilities),),j.int32)
    header=j.all(cap[:3]==j.array([BURN_STATUS,POISON_STATUS,len(data.quiet_abilities)],j.int32))
    return header&((status==BURN_STATUS)|(status==POISON_STATUS))&(counter==0)&(ability>0)&(ability<len(data.quiet_abilities))&(cap[3+j.clip(ability,0,len(data.quiet_abilities)-1)]==1)

def _ability_quiet(ability,status,counter,data,*,outgoing=False):
    result=data.quiet_abilities[j.clip(ability,0,len(data.quiet_abilities)-1)]
    dormant=getattr(data,'outgoing_dormant_abilities',None)
    if outgoing and dormant is not None:
        life._typed(dormant,data.quiet_abilities.shape,j.bool_)
        result |= (status==0)&(counter==0)&dormant[j.clip(ability,0,len(dormant)-1)]
    if outgoing:result |= _insomnia_status(ability,status,counter,data)
    return result

def _quiet(s,side,physical,data,*,outgoing_toxic=False,outgoing_paralysis=False,outgoing_substitute=False,outgoing_dormant=False):
    r=j.arange(len(side));seat=j.clip(side,0,1);mon=j.clip(physical,0,5);ability=s.ability[r,seat,mon];item=s.item[r,seat,mon]
    status=s.status[r,seat,mon];counter=s.status_counter[r,seat,mon]
    status_ok=(status==0)&(counter==0)
    if outgoing_toxic:status_ok |= (status==TOXIC_STATUS)&(counter>=0)&(counter<=15)
    # PAR has no switch hook. The actual before-move result is still `acts`;
    # only this outgoing source-bound path may retain PAR, never the recipient.
    if outgoing_paralysis:status_ok |= (status==PARALYSIS_STATUS)&(counter==0)
    if outgoing_dormant:status_ok |= _insomnia_status(ability,status,counter,data)
    good=(physical>=0)&(physical<6)&(s.hp[r,seat,mon]>0)&status_ok&(ability>0)&(ability<len(data.quiet_abilities))&_ability_quiet(ability,status,counter,data,outgoing=outgoing_dormant)&(item>=0)&(item<len(data.quiet_items))&data.quiet_items[j.clip(item,0,len(data.quiet_items)-1)]
    good &= (ability==s.base_ability[r,seat,mon])&j.all(s.types[r,seat,mon]==s.base_types[r,seat,mon],axis=1)&(s.spikes[r,seat]==0)&j.all(s.trap_source<0,axis=(1,2))
    sub=s.substitute_hp[r,seat,mon]
    good &= ((sub>=0)&(sub<=s.max_hp[r,seat,mon])&(s.max_hp[r,seat,mon]<=1_000_000)) if outgoing_substitute else sub==0
    for n in ('confusion_counter','volatile','protect_counter','choice_move','encore_move','encore_counter','charge_move','charge_counter'):good &= getattr(s,n)[r,seat,mon]==0
    return good

def missing_transaction(tape,side,requested):
    """No prior owning input is an entered UNKNOWN, never empty acceptance."""
    from . import engine_execution_completeness as ex
    if tape.execution is not None:
        tape,ticket=ex.open_attempt(tape,ex.Site.BATON_PASS,side,active=requested);c=ex._context(tape)
        tape=ex._put_context(tape,ex.unhandled(c.ledger,ticket,active=requested,tables=c.tables),c)
    return tape._replace(refusal=tape.refusal|j.where(requested,256,0)),None

def _start_diagnostics(tape,state,side,move,caller,data,*,acts,eligible,transaction=None,private=None,registered=None,data_bound=None,valid=None,prior=None):
    """Observe actual current start operands; never read back by admission."""
    from . import engine_outcome_diagnostics as diag
    c=getattr(tape,'execution',None);b=len(tape.count)
    if not diag._valid_baton_start(getattr(c,'diagnostics',None),b):return tape
    if getattr(side,'shape',None)!=(b,)or getattr(side,'dtype',None)!=j.dtype(j.int32):return tape
    r=j.arange(b);seat=j.clip(side,0,1);physical=state.active[r,seat];safe=j.clip(physical,0,5);known=(side>=0)&(side<2)&(physical>=0)&(physical<6);base=state if private is None else private
    def mon(source,name,tail=()):
        value=getattr(source,name,None)
        return None if getattr(value,'shape',None)!=(b,2,6)+tail else(value[r,seat,safe],known)
    values=dict(actor_side=side,actor_physical=physical,move=move,caller_move=caller,eligible=eligible,acts=acts,begin_valid=valid,registered_bound=registered,data_bound=data_bound,transaction_present=j.full(b,transaction is not None,bool),history_side=None if transaction is None else transaction.history.side)
    for name,field in [('hp','hp'),('maximum','max_hp'),('status','status'),('counter','status_counter'),('ability','ability'),('base_ability','base_ability'),('item','item'),('substitute_hp','substitute_hp')]:values[name]=mon(base,field)
    for name in('confusion_counter','volatile','protect_counter','choice_move','encore_move','encore_counter','charge_move','charge_counter'):values['volatile_uint32'if name=='volatile'else name]=mon(state,name)
    for field,prefix in [('types','type'),('base_types','base_type')]:
        value=mon(base,field,(2,))
        for i in range(2):values[prefix+str(i)]=None if value is None else(value[0][:,i],value[1])
    status=base.status[r,seat,safe];counter=base.status_counter[r,seat,safe];ability=base.ability[r,seat,safe]
    status_ok=((status==0)&(counter==0))|((status==TOXIC_STATUS)&(counter>=0)&(counter<=15))|((status==PARALYSIS_STATUS)&(counter==0))|_insomnia_status(ability,status,counter,data)
    values['quiet_status_lookup']=(status_ok,known)
    values.update(spikes=(state.spikes[r,seat],(side>=0)&(side<2)),any_trap_source=j.any(state.trap_source>=0,axis=(1,2)),pending_residual=state.pending_residual,any_forced_switch=j.any(state.forced_switch,axis=1),any_baton_pass=j.any(state.baton_pass,axis=1))
    for i in range(7):values['boost'+str(i)]=(state.boosts[r,seat,safe,i],known)
    pub=None if transaction is None else transaction.public
    for name,field in [('public_known','known'),('public_pending','pending'),('public_substitute','substitute')]:
        value=getattr(pub,field,None);values[name]=None if getattr(value,'shape',None)!=(b,2)else(value[r,seat],(side>=0)&(side<2))
    boost=getattr(pub,'boosts',None)
    for i in range(7):values['public_boost'+str(i)]=None if getattr(boost,'shape',None)!=(b,2,7)else(boost[r,seat,i],(side>=0)&(side<2))
    for field,ns in [('ability','abilities'),('item','items')]:
        table=getattr(data,'quiet_'+ns,None);value=mon(base,field)
        if value is not None and getattr(table,'ndim',None)==1 and len(table)>0:
            ids=value[0];lookup=table[j.clip(ids,0,len(table)-1)]
            if field=='ability':lookup=_ability_quiet(ids,base.status[r,seat,safe],base.status_counter[r,seat,safe],data,outgoing=True)
            values[field+'_namespace_size']=j.full(b,len(table),j.int32);values['quiet_'+field+'_lookup']=(lookup,value[1])
    before=tape if prior is None else prior;context=getattr(before,'execution',None)
    values.update(stream_id=before.stream_id,frame=None if context is None else context.ledger.frame,raw_count=before.count,turn=before.cursor.turn,phase=before.cursor.phase,generation0=before.cursor.generation[:,0],generation1=before.cursor.generation[:,1],prior_tape_refusal=before.refusal,prior_execution_refusal=None if context is None else context.ledger.refusal,prior_kernel_risk=None if context is None or context.risks is None else context.risks.carry.risk)
    requested=eligible&(move==data.move_id)&(data.move_id>0)
    return diag.record_baton_start_on_tape(tape,values,active=requested)

def begin_start(tape,state,side,move,caller,context,*,acts,eligible):
    from . import engine_execution_completeness as ex
    if context is None or context.data.baton_pass is None:return tape,None
    if tape.execution is None or tape.execution.baton_transaction is None:
        tape=_start_diagnostics(tape,state,side,move,caller,context.data.baton_pass,acts=acts,eligible=eligible)
        return missing_transaction(tape,side,eligible&(move==context.data.baton_pass.move_id)&(context.data.baton_pass.move_id>0))
    from . import engine_direct_hit_controls as direct
    d=context.data.baton_pass;registered=direct._registered_bound(tape,context);tr=tape.execution.baton_transaction;r=j.arange(len(side));seat=j.clip(side,0,1);physical=state.active[r,seat]
    requested=eligible&(move==d.move_id)&(d.move_id>0)
    data_bound=_bound(tape,d)
    good=registered&data_bound&(caller==move)&acts&~state.pending_residual&~j.any(state.forced_switch,axis=1)&(tr.history.side<0)&tr.public.known[r,seat]&~tr.public.pending[r,seat]
    good &= _quiet(state,side,physical,d,outgoing_toxic=True,outgoing_paralysis=True,outgoing_substitute=True,outgoing_dormant=True)&j.all(tr.public.boosts[r,seat]==state.boosts[r,seat,physical],axis=1)&~j.any(state.baton_pass,axis=1)
    good &= tr.public.substitute[r,seat]==(state.substitute_hp[r,seat,physical]>0)
    prior=tape
    tape,ticket=ex.open_attempt(tape,ex.Site.BATON_PASS,side,active=requested)
    h=History(tape.stream_id,ticket.frame,tape.cursor.turn,tape.cursor.generation,state.active,side,tape.cursor.public_slot[r,seat],j.zeros(len(side),bool),state.boosts,snapshot(state),tr.order,suspend(state))
    tape=_start_diagnostics(tape,state,side,move,caller,d,acts=acts,eligible=eligible,transaction=tr,private=h.private,registered=registered,data_bound=data_bound,valid=good,prior=prior)
    return tape,Pending(ticket,requested,good,side,physical,j.full_like(side,-1),h,snapshot(state),state.boosts)
def finish_start(tape,p,state,passed,secondary):
    if p is None:return tape
    from . import engine_execution_completeness as ex
    r=j.arange(len(p.side));seat=j.clip(p.side,0,1);good=p.valid&passed&state.baton_pass[r,seat]&equal(p.before,snapshot(state))&j.all(p.boosts==state.boosts,axis=(1,2,3))
    z=j.zeros_like(p.side);start=tape.count;h=p.history
    tape=public._append(tape,BATON_PASS,p.side,p.physical,j.full_like(z,_data(tape).move_id),z,(h.turn,h.frame,h.generation[r,seat],h.public_slot,secondary.astype(j.int32)),p.requested&good)
    c=ex._context(tape);ledger=ex.unhandled(c.ledger,p.ticket,active=p.requested&~good,tables=c.tables);ledger=ex.emitted(ledger,p.ticket,tape,active=p.requested&good,tables=c.tables)
    record=Record(good&p.requested,j.zeros_like(good),z,start,p.ticket.slot,p.ticket.serial,p.ticket.frame,p.side,p.physical,j.full_like(z,-1),secondary,h,p.before,p.boosts,state.boosts,snapshot(state))
    tr=c.baton_transaction;tr=tr._replace(history=rows(record.known,tr.history,h),record=rows(p.requested,tr.record,record))
    return ex._put_context(tape,ledger,c)._replace(execution=c._replace(ledger=ledger,baton_transaction=tr))
def finalize_request(tape,state):
    if tape.execution is None or tape.execution.baton_transaction is None:return tape
    tr=tape.execution.baton_transaction;rec=tr.record;h=rec.history;r=j.arange(len(h.side));side=j.clip(h.side,0,1)
    good=rec.known&(rec.code==START)&state.pending_residual&state.baton_pass[r,side]&state.forced_switch[r,side]&~state.forced_switch[r,1-side]&j.all(state.active==h.active,axis=1)&equal(h.private,snapshot(state))&j.all(h.boosts==state.boosts,axis=(1,2,3))
    h=h._replace(suspended=suspend(state))
    rec=rec._replace(history=rows(good,rec.history,h),finalized=j.where(rec.code==START,good,rec.finalized));tr=tr._replace(record=rec,history=rows(good,tr.history,h))
    return tape._replace(execution=tape.execution._replace(baton_transaction=tr))

def before_copy(tape,state,side,target,context,*,eligible):
    from . import engine_execution_completeness as ex
    if context is None or context.data.baton_pass is None:return tape,None
    if tape.execution is None or tape.execution.baton_transaction is None:return missing_transaction(tape,side,eligible&state.baton_pass[j.arange(len(side)),j.clip(side,0,1)])
    from . import engine_direct_hit_controls as direct
    registered=direct._registered_bound(tape,context)
    tr=tape.execution.baton_transaction;h=tr.history;d=context.data.baton_pass;r=j.arange(len(side));seat=j.clip(side,0,1);old=state.active[r,seat]
    requested=eligible&state.baton_pass[r,seat]
    good=registered&_bound(tape,d)&tr.request_ok&(side==h.side)&(target==tr.recipient)&(old==h.active[r,seat])&_quiet(state,side,old,d,outgoing_toxic=True,outgoing_paralysis=True,outgoing_substitute=True,outgoing_dormant=True)&_quiet(state,side,target,d)&equal(h.private,snapshot(state))&j.all(h.boosts==state.boosts,axis=(1,2,3))
    tape,ticket=ex.open_attempt(tape,ex.Site.BATON_PASS,side,active=requested)
    return tape,Pending(ticket,requested,good,side,old,target,h,snapshot(state),state.boosts)
def after_copy(tape,p,state):
    if p is None:return tape
    tr=tape.execution.baton_transaction;r=j.arange(len(p.side));side=j.clip(p.side,0,1);target=j.clip(p.incoming,0,5);expected=p.boosts.at[r,side,target].set(p.boosts[r,side,p.physical])
    good=p.valid&equal(copied_private(p.before,p.side,p.physical,p.incoming),snapshot(state))&j.all(state.boosts==expected,axis=(1,2,3));h=p.history;z=j.zeros_like(p.side);start=tape.count
    tape=public._append(tape,BATON_PASS,p.side,p.physical,p.incoming,z+INSTALL,(h.turn,h.frame,h.generation[r,side],h.public_slot,z),p.requested&good)
    rec=Record(good&p.requested,j.zeros_like(good),z+INSTALL,start,p.ticket.slot,p.ticket.serial,p.ticket.frame,p.side,p.physical,p.incoming,j.zeros_like(good),h,p.before,p.boosts,state.boosts,snapshot(state))
    return tape._replace(execution=tape.execution._replace(baton_transaction=tr._replace(record=rows(p.requested,tr.record,rec))))
def copied_private(before,side,outgoing,incoming):
    """Predict only original copy_to_recipient's already-existing Sub HP write."""
    r=j.arange(len(side));seat=j.clip(side,0,1);old=j.clip(outgoing,0,5);new=j.clip(incoming,0,5)
    return before._replace(substitute_hp=before.substitute_hp.at[r,seat,new].set(before.substitute_hp[r,seat,old]))

def departed_private(before,side,outgoing):
    """Same original entry.depart TOX-counter reset; no physical write here.

    Major status remains on the outgoing mon and never transfers. Original
    tox.onSwitchIn resets its own stage; the existing kernel clears that private
    clock on departure. This predicts only that actual already-existing write.
    """
    r=j.arange(len(side));seat=j.clip(side,0,1);old=j.clip(outgoing,0,5)
    counter=before.status_counter[r,seat,old]
    expected=j.where(before.status[r,seat,old]==TOXIC_STATUS,0,counter)
    return before._replace(status_counter=before.status_counter.at[r,seat,old].set(expected),substitute_hp=before.substitute_hp.at[r,seat,old].set(0))

def reveal_matches(tape,state,side,outgoing,incoming):
    if tape.execution is None or tape.execution.baton_transaction is None:return j.zeros_like(side,bool)
    tr=tape.execution.baton_transaction;rec=tr.record;h=rec.history;r=j.arange(len(side));seat=j.clip(side,0,1);old=j.clip(outgoing,0,5);new=j.clip(incoming,0,5)
    expected=rec.boost_after.at[r,seat,old].set(0)
    return rec.known&(rec.code==INSTALL)&(rec.side==side)&(rec.outgoing==outgoing)&(rec.incoming==incoming)&tr.request_ok&(tape.count==rec.start+1)&j.all(state.active.at[r,seat].set(old)==h.active,axis=1)&equal(departed_private(copied_private(rec.before,side,outgoing,incoming),side,outgoing),snapshot(state))&j.all(state.boosts==expected,axis=(1,2,3))
def finish_install(tape,state,current_side,*,eligible):
    if tape.execution is None or tape.execution.baton_transaction is None:return tape
    from . import engine_execution_completeness as ex
    tr=tape.execution.baton_transaction;rec=tr.record;r=j.arange(len(tape.count));side=j.clip(rec.side,0,1);at=j.clip(rec.start+1,0,tape.words.shape[1]-1);raw=tape.words[r,at]
    good=rec.known&(rec.code==INSTALL)&(tape.count==rec.start+2)&(raw[:,0]==life.SWITCH_INSTALLED)&(raw[:,1]==rec.side)&(raw[:,2]==rec.outgoing)&(raw[:,3]==rec.side)&(raw[:,4]==rec.incoming)&(raw[:,6]==1)&j.all(raw[:,7:]==0,axis=1)
    good &= (tape.cursor.physical_active[r,side]==rec.incoming)&(tape.cursor.generation[r,side]==rec.history.generation[r,side]+1)&tape.cursor.established[r,side]
    expected=rec.boost_after.at[r,side,j.clip(rec.outgoing,0,5)].set(0)
    good &= equal(departed_private(copied_private(rec.before,rec.side,rec.outgoing,rec.incoming),rec.side,rec.outgoing),snapshot(state))&j.all(expected==state.boosts,axis=(1,2,3))
    ticket=ex.Ticket(tape.stream_id,tape.execution.tables.inventory_binding,rec.frame,rec.slot,rec.serial);c=ex._context(tape)
    requested=(rec.code==INSTALL)&eligible&(rec.side==current_side);ledger=ex.unhandled(c.ledger,ticket,active=requested&~good,tables=c.tables);ledger=ex.emitted(ledger,ticket,tape,active=requested&good,tables=c.tables)
    rec=rec._replace(finalized=j.where(requested,good,rec.finalized),boost_after=rows(requested,rec.boost_after,state.boosts),after=rows(requested,rec.after,snapshot(state)));h=tr.history._replace(side=j.where(requested&good,-1,tr.history.side),ready=tr.history.ready&~(requested&good))
    return tape._replace(execution=c._replace(ledger=ledger,baton_transaction=tr._replace(record=rec,history=h)))

def owned(tape,index,code=None):
    """Independent SAME claimed span and all cached copy/source operands."""
    from . import engine_execution_completeness as ex
    b=len(tape.count);r=j.arange(b);at=j.broadcast_to(j.asarray(index,j.int32),(b,))
    if tape.execution is None or tape.execution.baton_transaction is None:return j.zeros(b,bool)
    tr=tape.execution.baton_transaction;rec=tr.record;h=rec.history;history_shape(h,b);d=_data(tape);side=j.clip(rec.side,0,1);old=j.clip(rec.outgoing,0,5);new=j.clip(rec.incoming,0,5)
    history_shape(h._replace(private=rec.before),b);history_shape(h._replace(private=rec.after),b)
    ledger=tape.execution.ledger;start=rec.start;pos=j.clip(start,0,tape.words.shape[1]-1);raw=tape.words[r,pos];length=j.where(rec.code==START,1,2)
    claim=ledger.claimed[r,pos];slot=j.clip(claim-1,0,ledger.site.shape[1]-1)
    good=_bound(tape,d)&rec.known&rec.finalized&(rec.frame==ledger.frame)&(rec.side>=0)&(rec.side<2)&(rec.outgoing>=0)&(rec.outgoing<6)&(rec.frame==tape.execution.ledger.frame)
    good &= (start>=0)&(start+length<=tape.count)&(at>=start)&(at<start+length)&(claim==rec.slot+1)&(claim>0)&(claim<=ledger.count)&(ledger.site[r,slot]==int(ex.Site.BATON_PASS))&(ledger.side[r,slot]==rec.side)&(ledger.serial[r,slot]==rec.serial)&(ledger.stage[r,slot]==2)&(ledger.start[r,slot]==start)&(ledger.end[r,slot]==start+length)
    expected=j.stack((j.full(b,BATON_PASS,j.int32),rec.side,rec.outgoing,j.where(rec.code==START,d.move_id,rec.incoming),rec.code,h.turn,h.frame,h.generation[r,side],h.public_slot,j.where(rec.code==START,rec.secondary.astype(j.int32),0)),axis=1)
    good &= j.all(raw==expected,axis=1)&((rec.code==START)|(rec.code==INSTALL))&((raw[:,9]==0)|(raw[:,9]==1))
    for n in range(2):
        p=j.clip(start+n,0,tape.words.shape[1]-1);used=n<length
        valid=(ledger.claimed[r,p]==claim)&j.all(ledger.words[r,p]==tape.words[r,p],axis=1)&(ledger.event_turn[r,p]==tape.event_turn[r,p])&(ledger.event_phase[r,p]==tape.event_phase[r,p])&j.all(ledger.event_generation[r,p]==tape.event_generation[r,p],axis=1)
        valid &= (tape.event_turn[r,p]==ledger.turn[r,slot])&(tape.event_phase[r,p]==ledger.phase[r,slot])&j.all(tape.event_generation[r,p]==ledger.generation[r,slot],axis=1)
        good &= ~used|valid
    prefix=j.clip(start-1,0,tape.words.shape[1]-1);decl=tape.words[r,prefix]
    declared=(start>0)&(decl[:,0]==public.MOVE)&(decl[:,1]==rec.side)&(decl[:,2]==rec.outgoing)&(decl[:,3]==1-rec.side)&(decl[:,5]==d.move_id)&j.all(decl[:,6:]==0,axis=1)
    declared &= (tape.event_turn[r,prefix]==h.turn)&j.all(tape.event_generation[r,prefix]==h.generation,axis=1)
    good &= j.where(rec.code==START,declared&(h.frame==rec.frame)&equal(rec.before,rec.after)&equal(h.private,rec.before)&j.all(h.boosts==rec.boost_before,axis=(1,2,3)),True)
    installed=tape.words[r,j.clip(start+1,0,tape.words.shape[1]-1)]
    expected_after=h.boosts.at[r,side,new].set(h.boosts[r,side,old]);expected_after=expected_after.at[r,side,old].set(0)
    transfer=(rec.incoming>=0)&(rec.incoming<6)&(rec.incoming!=rec.outgoing)&h.ready&(h.frame+1==rec.frame)&tr.request_ok&(tr.recipient==rec.incoming)&tr.public.known[r,side]&tr.public.pending[r,side]
    transfer &= equal(departed_private(copied_private(rec.before,rec.side,rec.outgoing,rec.incoming),rec.side,rec.outgoing),rec.after)&(tr.public.substitute[r,side]==(rec.before.substitute_hp[r,side,old]>0))
    transfer &= j.all(tr.order==h.order,axis=(1,2))&equal(h.private,rec.before)&j.all(h.boosts==rec.boost_before,axis=(1,2,3))&j.all(expected_after==rec.boost_after,axis=(1,2,3))
    transfer &= (installed[:,0]==life.SWITCH_INSTALLED)&(installed[:,1]==rec.side)&(installed[:,2]==rec.outgoing)&(installed[:,3]==rec.side)&(installed[:,4]==rec.incoming)&(installed[:,5]==1)&(installed[:,6]==1)&j.all(installed[:,7:]==0,axis=1)
    details=j.stack(tuple(getattr(rec.before,n)[r,side,new]for n in ('species','level','gender','hp','max_hp','status')),axis=1);transfer &= j.all(tape.switch_details[r,j.clip(start+1,0,tape.words.shape[1]-1)]==details,axis=1)
    good &= j.where(rec.code==INSTALL,transfer,True)
    if code is not None:good &= rec.code==code
    return good

def public_guard(tape,pub):
    """Read the actual supplied before-public, never trust an acceptance flag."""
    b=len(tape.count);r=j.arange(b)
    if tape.execution is None or tape.execution.baton_transaction is None:return j.ones(b,bool)
    rec=tape.execution.baton_transaction.record;h=rec.history;side=j.clip(rec.side,0,1);old=j.clip(rec.outgoing,0,5)
    present=j.any((j.arange(tape.words.shape[1])[None,:]<tape.count[:,None])&(tape.words[:,:,0]==BATON_PASS),axis=1)
    same=(pub.cursor.stream_id==tape.stream_id)&(pub.cursor.turn==h.turn)&j.all(pub.mons.active.at[r,side].set(h.public_slot)==tape.cursor.public_slot.at[r,side].set(h.public_slot),axis=1)
    # On installation the actual before public is still the outgoing encounter.
    sub,quiet=public_substitute(pub)
    same &= (pub.mons.active[r,side]==h.public_slot)&j.all(pub.mons.boosts[r,side]==h.boosts[r,side,old],axis=1)&quiet[r,side]&(sub[r,side]==(h.private.substitute_hp[r,side,old]>0))
    same &= j.where(rec.code==INSTALL,pub.context.tendency.baton[r,side],~pub.context.tendency.baton[r,side])
    return ~present|same

def mature(history,frame,ledgers,publics,requests):
    """Only actual BOTH-owner stage3 consumption + real force/wait can persist."""
    if history is None:return None
    tr=frame.callbacks.execution.baton_transaction
    if tr is None:return history
    rec=tr.record;r=j.arange(len(history.side));side=j.clip(history.side,0,1);history_shape(history,len(history.side));good=owned(frame.callbacks,rec.start,START)&equal(history,tr.history)&equal(history,rec.history)
    for ledger in ledgers:
        slot=j.clip(rec.slot,0,ledger.site.shape[1]-1)
        good &= (ledger.stream_id==history.stream_id)&(ledger.refusal==0)&(ledger.stage[r,slot]==3)&(ledger.serial[r,slot]==rec.serial)&(ledger.site[r,slot]==39)
    from . import private_requests as req
    for owner,(pub,q)in enumerate(zip(publics,requests,strict=True)):
        packet=q.actions.facts.request;own=history.side==owner
        good &= q.actions.admitted&j.where(own,packet.kind==int(req.RequestKind.FORCE_SWITCH),packet.kind==int(req.RequestKind.WAIT))
        sub,quiet=public_substitute(pub)
        good &= quiet[r,side]&(sub[r,side]==(history.private.substitute_hp[r,side,history.active[r,side]]>0))
        good &= (pub.cursor.stream_id==history.stream_id)&pub.context.tendency.baton[r,side]&j.all(pub.mons.boosts[r,side]==history.boosts[r,side,history.active[r,side]],axis=1)
    return history._replace(ready=j.where(rec.code==START,good,history.ready))
