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
    return_move: object = None # canonical source-bound dynamic Return only
    seismic_move: object = None # original fixed-LEVEL source, literal power0
    hidden_power_moves: object = None # installed original aliases, actual member operands required
    double_edge_move: object = None # canonical source-bound Sub damage + separate recoil only


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
    actor_happiness: object = None
    caller_move: object = None
    actor_level: object = None
    actor_hp_type: object = None
    actor_hp_power: object = None
    resolved_type: object = None


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
const rows=m.vocabulary.moves.map(id=>{const x=d.moves.get(id);return {id:x.id,exists:x.exists,gen:x.gen,target:x.target,power:x.basePower,type:x.type,
 damage:x.damage||null,recoil:x.recoil||null,accuracy:x.accuracy,priority:x.priority,recoilExotic:!!(x.multihit||x.damage||x.drain||x.heal||x.self||x.secondary||x.secondaries||x.boosts||x.status||x.volatileStatus||x.sideCondition||x.slotCondition||x.condition||x.weather||x.terrain||x.pseudoWeather||x.selfdestruct||x.forceSwitch||x.selfSwitch||x.ohko),fixedExotic:!!(x.multihit||x.recoil||x.drain||x.heal||x.self||x.secondary||x.secondaries||x.boosts||x.status||x.volatileStatus||x.sideCondition||x.slotCondition||x.condition||x.weather||x.terrain||x.pseudoWeather||x.selfdestruct||x.forceSwitch||x.selfSwitch||x.ohko),returnCallback:x.id==='return'&&x.basePowerCallback?require('crypto').createHash('sha256').update(x.basePowerCallback.toString()).digest('hex'):null,category:x.category,bypass:!!x.flags.bypasssub,callbacks:Object.keys(x).filter(k=>(/^on/.test(k)||/Callback$/.test(k))&&x[k]!==undefined),
 exotic:!!(x.multihit||x.damage||x.recoil||x.drain||x.heal||x.self||x.secondary?.self||(Array.isArray(x.secondaries)&&x.secondaries.some(v=>v.self))||x.boosts||x.status||x.volatileStatus||x.sideCondition||x.slotCondition||x.condition||x.weather||x.terrain||x.pseudoWeather||x.selfdestruct||x.forceSwitch||x.selfSwitch||x.ohko)}});
process.stdout.write(JSON.stringify({rows,hiddenPower:Object.fromEntries(['onModifyType','basePowerCallback','onModifyMove'].map(k=>[k,require('crypto').createHash('sha256').update(d.moves.get('hiddenpower')[k].toString()).digest('hex')])),getDamage:require('crypto').createHash('sha256').update(require(path.join(root,'dist/sim/battle-actions.js')).BattleActions.prototype.getDamage.toString()).digest('hex'),substitute:{hit:sub.onTryPrimaryHit.toString(),end:sub.onEnd.toString()},source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));
"""
    run=subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],
        input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30)
    source=json.loads(run.stdout)
    if (json.dumps(source['source_tree'],sort_keys=True)!=origin.source_tree_json
            or json.dumps(source['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json):
        raise ValueError('original Substitute source changed')
    table=catalog.engine_table
    mask=np.zeros(len(table.power),bool);return_move=0;seismic_move=0;double_edge_move=0;hidden=np.zeros(len(table.power),bool)
    for i,(name,row) in enumerate(zip(metadata['vocabulary']['moves'],source['rows'],strict=True)):
        mask[i]=bool(i>0 and name==row['id'] and row['exists'] and row['gen']<=3
            and row['target'] in ('normal','allAdjacent','allAdjacentFoes') and row['category'] in ('Physical','Special')
            and row['power']>0 and not row['callbacks'] and not row['exotic'] and not row['bypass']
            and int(table.power[i])==row['power'] and catalog.ids[i]>0 and catalog.targets[i]==1)
        if name=='doubleedge' and i>0 and catalog.ids[i]>0:
            if (double_edge_move or row['id']!='doubleedge' or not row['exists'] or row['gen']!=1
                or row['target']!='normal' or row['category']!='Physical' or row['type']!='Normal'
                or row['power']!=120 or row['recoil']!=[1,3] or row['callbacks'] or row['bypass'] or row['recoilExotic']
                or row['accuracy']!=100 or row['priority']!=0 or not bool(table.physical[i]) or int(table.type[i])!=metadata['vocabulary']['types'].index('normal')
                or int(table.power[i])!=120 or int(table.accuracy[i])!=100 or bool(table.always_hit[i])
                or int(table.priority[i])!=0 or catalog.targets[i]!=1 or table.damage_moves is None
                or int(table.damage_moves.recoil_den[i])!=3 or bool(table.damage_moves.struggle[i])):
                raise ValueError('canonical original Double-Edge Substitute source required')
            if hashlib.sha256(source['substitute']['hit'].encode()).hexdigest()!='050e1c1fbdb80ea3ab7bc63909eb72433cb45e9c49ff7889e16ec2c207528a24':
                raise ValueError('original Substitute damage-before-recoil callback required')
            double_edge_move=i;mask[i]=True
    if table.variable_moves is not None:
        installed=np.asarray(table.variable_moves.returns);wanted=np.asarray([n=='return'for n in metadata['vocabulary']['moves']],bool)
        if not np.array_equal(installed,wanted):raise ValueError('original installed Return resolver required')
        for i in np.flatnonzero(wanted):
            row=source['rows'][int(i)]
            if (i<=0 or row['id']!='return' or not row['exists'] or row['gen']>3 or row['target']!='normal' or row['category']!='Physical' or row['power']!=0 or row['callbacks']!=['basePowerCallback'] or row['returnCallback']!='a2bce14a77c22d55d42a7e1bd68c59689ef9a08d57dbabb645afb96a6ac2ecac' or row['exotic'] or row['bypass'] or int(table.power[i])!=0 or catalog.ids[i]<=0 or catalog.targets[i]!=1):raise ValueError('canonical original Return Substitute source required')
            if return_move:raise ValueError('unique canonical Return row required')
            return_move=int(i);mask[i]=True
    if table.damage_moves is not None:
        for i,(name,row)in enumerate(zip(metadata['vocabulary']['moves'],source['rows'],strict=True)):
            if name!='seismictoss' or i==0 or catalog.ids[i]<=0:continue
            dm=table.damage_moves
            if (seismic_move or row['id']!='seismictoss' or not row['exists'] or row['gen']>3 or row['target']!='normal' or row['category']!='Physical' or row['power']!=0 or row['damage']!='level' or row['callbacks'] or row['fixedExotic'] or row['bypass'] or int(table.power[i])!=0 or int(table.type[i])!=2 or not bool(table.physical[i]) or int(table.accuracy[i])!=100 or bool(table.always_hit[i]) or int(table.priority[i])!=0 or not bool(dm.level_damage[i]) or int(dm.constant_damage[i])!=0 or int(dm.recoil_den[i])!=0 or bool(dm.struggle[i]) or catalog.targets[i]!=1):raise ValueError('canonical original fixed-LEVEL Seismic Toss Substitute source required')
            if source['getDamage']!='dfd031900da8c77e5e1a2721858d8d4a7804bc7def1b7a1d486220521a6a7f12' or hashlib.sha256(source['substitute']['hit'].encode()).hexdigest()!='050e1c1fbdb80ea3ab7bc63909eb72433cb45e9c49ff7889e16ec2c207528a24':raise ValueError('original fixed-LEVEL Substitute source changed')
            seismic_move=i;mask[i]=True
    if table.variable_moves is not None:
        names=metadata['vocabulary']['moves'];types=metadata['vocabulary']['types'];vm=table.variable_moves
        expected=np.asarray([n.startswith('hiddenpower')for n in names],bool)
        if not np.array_equal(np.asarray(vm.hidden_power),expected):raise ValueError('original installed Hidden Power resolver required')
        if not np.array_equal(np.asarray(vm.special_type),np.asarray([n in {'fire','water','grass','ice','electric','dark','psychic','dragon'}for n in types],bool)):raise ValueError('original Gen3 Hidden Power category selector required')
        if source['hiddenPower']!={'onModifyType':'972b0d3cbdfee6a71f24de8909efddb94ea206f36e6bf30dbc60c50824e4cb9e','basePowerCallback':'b55dccff16e0dca753a95388f4f20c13801ad030bbe9e545733c1b20c67b72d0','onModifyMove':'997a83bf939c8589061b786f7d5d5045b43716f502837b666be7e6c21c104b08'}:raise ValueError('original canonical Hidden Power member callbacks changed')
        public_hidden=catalog.public_moves.ids.get('hiddenpower',0)
        for i in np.flatnonzero(expected):
            row=source['rows'][int(i)];canonical=names[i]=='hiddenpower'
            if i<=0 or catalog.ids[i]<=0:continue
            if (row['id']!='hiddenpower' or not row['exists'] or row['gen']>3 or row['target']!='normal' or row['category']not in ('Physical','Special') or row['fixedExotic'] or row['damage']is not None or row['bypass']
                or row['callbacks']!=(['onModifyType','basePowerCallback','onModifyMove']if canonical else[]) or row['power']!=(0 if canonical else 70)
                or int(table.power[i])!=row['power'] or int(table.type[i])!=types.index(row['type'].lower()) or bool(table.physical[i])!=(row['category']=='Physical')
                or int(table.accuracy[i])!=100 or bool(table.always_hit[i]) or int(table.priority[i])!=0 or catalog.targets[i]!=1 or public_hidden<=0 or catalog.ids[i]!=public_hidden):raise ValueError('canonical original installed Hidden Power/alias Substitute source required')
            hidden[i]=True;mask[i]=True
    payload=dict(source=source,candidates=mask.tolist(),return_move=return_move,seismic_move=seismic_move,double_edge_move=double_edge_move,hidden_power_moves=hidden.tolist(),engine=catalog.engine_binding,
        producer=catalog.binding,helper=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    words=lambda x:j.asarray(x,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True,default=lambda x:np.asarray(x).tolist()).encode()).digest(),dtype='>u4').astype(np.uint32)),
        words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),
        words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(mask),j.asarray(return_move,j.int32),j.asarray(seismic_move,j.int32),j.asarray(hidden),j.asarray(double_edge_move,j.int32))


def shape(data):
    if type(data) is not Data:raise ValueError('typed original Substitute damage data required')
    for x in data[:5]:life._typed(x,(8,),j.uint32)
    life._typed(data.candidates,(len(data.candidates),),j.bool_)
    life._typed(data.return_move,(),j.int32);life._typed(data.seismic_move,(),j.int32);life._typed(data.hidden_power_moves,data.candidates.shape,j.bool_)
    if data.double_edge_move is not None:life._typed(data.double_edge_move,(),j.int32)


def _double_edge(context,move,power,caller):
    data=context.data.substitute_damage
    required=j.zeros_like(move,bool)if data.double_edge_move is None else(data.double_edge_move>0)&(move==data.double_edge_move)
    valid=j.zeros_like(move,bool)
    if caller is not None:
        life._typed(caller,move.shape,j.int32)
        valid=(caller==move)&(power==120)
    return ~required|valid


def _return_power(context,move,power,happiness,caller_move):
    """Only actual owning call operands, never default happiness or called IDs."""
    data=context.data.substitute_damage;shape(data)
    required=(data.return_move>0)&(move==data.return_move)
    good=j.zeros_like(move,bool)
    if happiness is not None and caller_move is not None:
        b=len(move)
        for x in (power,happiness,caller_move):life._typed(x,(b,),j.int32)
        life._typed(context.data.return_ko_move,(),j.int32)
        good=(data.return_move==context.data.return_ko_move)&(happiness>=0)&(happiness<=255)&(caller_move==move)&(power==j.maximum(1,happiness*10//25))
    return ~required|good



def _fixed(context,move):
    data=context.data.substitute_damage;shape(data)
    from . import engine_direct_hit_controls as direct
    return (data.seismic_move>0)&(move==data.seismic_move)&direct.fixed_level_move_device(context.data,move)


def _source_power(context,move,power,level,caller):
    """Literal zero power and current same-call level; no default level."""
    data=context.data.substitute_damage;shape(data)
    required=(data.seismic_move>0)&(move==data.seismic_move)
    valid=j.zeros_like(move,bool)
    if level is not None and caller is not None:
        b=len(move)
        for x in(power,level,caller):life._typed(x,(b,),j.int32)
        valid=_fixed(context,move)&(power==0)&(level>0)&(level<=100)&(caller==move)
    return j.where(required,valid,power>0)

def _hidden_power(context,move,power,hp_type,hp_power,resolved_type,caller):
    data=context.data.substitute_damage;shape(data);b=len(move)
    required=(move>0)&(move<len(data.hidden_power_moves))&data.hidden_power_moves[j.clip(move,0,len(data.hidden_power_moves)-1)]
    good=j.zeros(b,bool)
    if hp_type is not None and hp_power is not None and resolved_type is not None and caller is not None:
        for x in(power,hp_type,hp_power,resolved_type,caller):life._typed(x,(b,),j.int32)
        good=(hp_type>0)&(hp_type<18)&(hp_power>=30)&(hp_power<=70)&(power==hp_power)&(resolved_type==hp_type)&(caller==move)
    return ~required|good


def route(context,move,*,acts,power,hit,immune,protected,enduring,bypass,absorbed,sub_hp,actor_happiness=None,caller_move=None,actor_level=None,actor_hp_type=None,actor_hp_power=None,resolved_type=None):
    if context is None or context.data.substitute_damage is None:return j.zeros_like(acts)
    data=context.data.substitute_damage;shape(data)
    valid=(move>0)&(move<len(data.candidates))
    return (_double_edge(context,move,power,caller_move)&_return_power(context,move,power,actor_happiness,caller_move)&_hidden_power(context,move,power,actor_hp_type,actor_hp_power,resolved_type,caller_move)&acts&valid&data.candidates[j.clip(move,0,len(data.candidates)-1)]&_source_power(context,move,power,actor_level,caller_move)
        &hit&~immune&~protected&~enduring&~bypass&~absorbed&(sub_hp>0))


def valid_write(write,b,*,pending=None):
    if type(write) is not Write:raise ValueError('literal Substitute damage Write required')
    for i,x in enumerate(write):life._typed(x,(b,),j.bool_ if 1<=i<=7 else j.int32)
    from .engine_direct_hit_controls import _positive_ratio
    source=write.power>0
    if pending is not None:
        source=_source_power(pending.context,pending.move,write.power,pending.actor_level,pending.caller_move)
        source &= _hidden_power(pending.context,pending.move,write.power,pending.actor_hp_type,pending.actor_hp_power,pending.resolved_type,pending.caller_move)
        fixed=_fixed(pending.context,pending.move)
        if pending.actor_level is None:source &= ~fixed
        else:source &= ~fixed|(write.damage==pending.actor_level)
    return (source&write.hit&~write.immune&~write.protected&~write.enduring
        &~write.bypass&~write.absorbed&_positive_ratio(write.numerator,write.denominator)
        &(write.old_sub>0)&(write.old_sub<=1_000_000)&(write.damage>0)
        &(write.dealt==j.minimum(write.old_sub,write.damage))&(write.dealt>0)
        &(write.new_sub==write.old_sub-write.dealt)&(write.new_sub>=0)
        &(write.old_hp>0)&(write.new_hp==write.old_hp))


def raw_words(p,write):
    fixed=_fixed(p.context,p.move)
    effect=j.where(fixed,0,j.where(write.numerator>write.denominator,1,j.where(write.numerator<write.denominator,-1,0)))
    return j.stack((j.full_like(p.actor,SUBSTITUTE_DAMAGE),p.actor,p.physical,1-p.actor,p.target,
        p.move,write.old_sub,write.new_sub,(write.critical&~fixed).astype(j.int32),effect),axis=1)


def validate_raw(raw):
    return ((raw[:,0]==SUBSTITUTE_DAMAGE)&(raw[:,1]>=0)&(raw[:,1]<2)
        &(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,3]==1-raw[:,1])&(raw[:,4]>=0)&(raw[:,4]<6)
        &(raw[:,5]>0)&(raw[:,6]>raw[:,7])&(raw[:,7]>=0)&(raw[:,6]<=1_000_000)
        &((raw[:,8]==0)|(raw[:,8]==1))&(raw[:,9]>=-1)&(raw[:,9]<=1))


def _declaration(tape,index,raw,cursor,turn,phase,generation):
    """Same source MOVE/stamps at capture, outcome and eventual lower drain."""
    b=len(tape.count);rows=j.arange(b);index=j.broadcast_to(j.asarray(index,j.int32),(b,))
    at=j.clip(index,0,tape.words.shape[1]-1);prefix=j.clip(index-1,0,tape.words.shape[1]-1)
    decl=tape.words[rows,prefix];actor=j.clip(raw[:,1],0,1)
    good=(life._valid(cursor)&(index>0)&(decl[:,0]==public.MOVE)
        &j.all(decl[:,1:6]==raw[:,1:6],axis=1)&j.all(decl[:,6:]==0,axis=1)
        &(raw[:,2]==cursor.physical_active[rows,actor])&(raw[:,4]==cursor.physical_active[rows,1-actor])
        &(cursor.turn==turn)&(cursor.phase==phase)&j.all(cursor.generation==generation,axis=1))
    for position in (prefix,at):
        good &= ((tape.event_turn[rows,position]==turn)&(tape.event_phase[rows,position]==phase)
            &j.all(tape.event_generation[rows,position]==generation,axis=1)
            &j.all(tape.switch_details[rows,position]==0,axis=1))
    return good


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
    start=ledger.start[rows,slot];at=j.clip(start,0,tape.words.shape[1]-1)
    raw=tape.words[rows,at]
    good=(bound&same&valid_write(write,b,pending=p)&_double_edge(p.context,p.move,write.power,p.caller_move)&_return_power(p.context,p.move,write.power,p.actor_happiness,p.caller_move)&p.requested&(p.move>0)&(p.move<len(data.candidates))
        &data.candidates[j.clip(p.move,0,len(data.candidates)-1)]
        &(ledger.site[rows,slot]==int(ex.Site.SUBSTITUTE_DAMAGE))&(ledger.side[rows,slot]==1-p.actor)
        &(start>0)&(tape.count==start+1)&validate_raw(raw)&j.all(raw==raw_words(p,write),axis=1)
        &_declaration(tape,start,raw,tape.cursor,ledger.turn[rows,slot],ledger.phase[rows,slot],ledger.generation[rows,slot]))
    return good


def record(tape,p,write):
    from . import engine_execution_completeness as ex
    raw=raw_words(p,write);before=tape
    tape=public._append(tape,SUBSTITUTE_DAMAGE,p.actor,p.physical,1-p.actor,p.target,
        tuple(raw[:,i]for i in range(5,10)),p.requested&valid_write(write,len(tape.count),pending=p)&_return_power(p.context,p.move,write.power,p.actor_happiness,p.caller_move))
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
    if actual.risks is not None:
        fixed=(actual.risks.tables.seismic_toss_move>0)&(raw[:,5]==actual.risks.tables.seismic_toss_move)
        same &= ~fixed|((raw[:,8]==0)&(raw[:,9]==0))
    return same&_declaration(tape,index,raw,current,ledger.turn[rows,slot],ledger.phase[rows,slot],ledger.generation[rows,slot])
