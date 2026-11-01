"""Original writer + SAME-operation inert components, no engine turns/models."""
import ast,json,subprocess,unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_self_boost_callbacks as cb,engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers
from gpu_engine import transition as tr,public_belief,v4_encoder as v4,public_wire,state
from gpu_engine import engine_source_frame_witness as witness
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_hidden_power_substitute_input as fixtures
import test_gpu_engine_transition_observer_wiring as static
BASE='afc80dc70525b7eb8723233015ce6495614bef08'
class SelfBoostInputTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.HiddenPowerSubstituteInputTests.setUpClass();f=fixtures.HiddenPowerSubstituteInputTests;cls.f=f;cls.metadata=deepcopy(f.metadata)
  dex=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves);cls.names=('tailglow','agility','dragondance','surf')
  flags=json.loads(subprocess.check_output(['node','-e',"const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3');console.log(JSON.stringify(JSON.parse(process.argv[2]).map(n=>d.moves.get(n).flags)))",str(showdown_root()),json.dumps(cls.names)],text=True))
  for name,flag in zip(cls.names,flags):
   info=dex.move_info(name);moves[name]=info
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   i=cls.metadata['vocabulary']['moves'].index(name);cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=True if name in ('tailglow','agility','dragondance') else int(info.accuracy),priority=info.priority,flags=flag)
  cls.dex=ShowdownDex(moves,f.dex.species,f.dex.type_chart)
  variants={name:Gen3RandbatSpeciesUniverse(name,80,(Gen3RandbatVariant('boost-'+name,'unit',name,'unit',80,cls.names,'Pressure','Leftovers'),))for name in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-selfboost-component'),variants)
  labels=set(f.catalog.vocabulary.tokens)
  for n in cls.names:labels|={'move:'+n,'belief:possible_move:'+n}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=f.catalog.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  cls.rules=request.load_source_rules(showdown_root(),cls.metadata);req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer);facts=own.compile_own_facts(req,cls.catalog)
  cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.prepared=driver.prepare(req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.asarray(cls.producer.engine_binding,np.uint32))
  cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))
  cls.consume=staticmethod(jax.jit(lambda l,f,s,p:ex.consume_frame_device(l,f,s,p,active=j.arange(8)<6,tables=cls.runtime['execution_compiled'].device,adapter_tables=cls.runtime['public'].device,catalog=cls.catalog)))
 def test_cold_import_orders(self):
  import os,sys
  for names in [('gpu_engine.benchmark_retained_games','gpu_engine.engine_self_boost_callbacks','gpu_engine.engine_full_v4_event_device'),('gpu_engine.engine_full_v4_event_device','gpu_engine.engine_lifecycle_event_device','gpu_engine.engine_self_boost_callbacks','gpu_engine.benchmark_retained_games')]:
   result=subprocess.run([sys.executable,'-B','-c','import importlib;'+''.join('importlib.import_module('+repr(n)+');'for n in names)],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30)
   self.assertEqual(result.returncode,0,result.stderr)
 def component(self):
  b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);actor=rows%2;active=j.stack((z+3,z+4),axis=1)
  s=state.empty_state(b)._replace(active=active,gender=j.ones((b,2,6),j.int32),hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),level=j.full((b,2,6),80,j.int32),done=~yes)
  cursor=life.register(rows+188001,**self.kw);initial=life._tape(cursor,self.producer,64)
  for side in (0,1):initial=life._reveal(initial,s,z+side,z-1,active[:,side],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes)
  tape=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];rc=self.runtime['kernel_risk_compiled'];tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes)
  carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,rc.device),tape,rc.device,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,rc.device)),refusal=z+1)
  names=('tailglow','agility','dragondance','tailglow','agility','dragondance','tailglow','agility');move=j.array([self.metadata['vocabulary']['moves'].index(n)for n in names],j.int32);physical=active[rows,actor]
  tape=public.move(tape,actor,physical,active[rows,1-actor],move,eligible=yes);old=j.zeros((b,7),j.int32).at[6,2].set(5);caller=move.at[7].set(self.metadata['vocabulary']['moves'].index('sleeptalk'))
  before=tape;tape,p=cb.begin(tape,s,actor,move,caller,old,self.runtime['direct_hit_context'],self.table.effects,eligible=yes)
  new=j.clip(old+self.table.effects.self_boosts[move],-6,6);tape,c=cb.finish(tape,p,new,yes)
  return SimpleNamespace(state=s,cursor=cursor,initial=initial,before=before,tape=tape,capture=c,actor=actor,move=move,old=old,new=new)
 def test_original_source_default_actual_call_operands_and_legacy(self):
  witness._source();self.assertEqual(len(witness.SOURCE_PINS),79)
  d=self.helper.device.self_boost
  for name,stats,deltas in [('tailglow',[2,-1],[2,0]),('agility',[4,-1],[2,0]),('dragondance',[0,4],[1,1])]:
   i=self.metadata['vocabulary']['moves'].index(name);self.assertEqual(d.stats[i].tolist(),stats);self.assertEqual(d.deltas[i].tolist(),deltas);self.assertEqual(int(self.runtime['kernel_risk_compiled'].device.selected_risk[i]),0)
  self.assertIsNone(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.self_boost)
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));after=ast.parse(Path('gpu_engine/transition.py').read_text());baseline=dump_default_program(old);self.assertEqual(dump_default_program(after),baseline);assert_hp_rng_mutations_detected(self,after,baseline)
  self.assertEqual(static._undefined_step_globals(Path('gpu_engine/transition.py').read_text()),set())
  step=static._step(Path('gpu_engine/transition.py').read_text());calls={x.func.attr:x for x in ast.walk(step)if isinstance(x,ast.Call)and isinstance(x.func,ast.Attribute)and isinstance(x.func.value,ast.Name)and x.func.value.id=='self_boost_callbacks'}
  sentinel=object();env=dict(_public_trace=sentinel,work=sentinel,side=sentinel,move=sentinel,caller_move=sentinel,boosts=sentinel,_direct_hit_context=sentinel,_direct_hit_registered=True,effects=sentinel,caller_acts=True,acts=True,hit=True,immune=False,_self_boost_pending=sentinel,self_boost_callbacks=SimpleNamespace(begin=lambda *a,**kw:(a,kw),finish=lambda *a:(a,{})))
  args,kw=eval(compile(ast.Expression(calls['begin']),'actual-transition','eval'),{},env);self.assertEqual(len(args),8);self.assertIs(args[4],sentinel);self.assertIs(args[5],sentinel);self.assertIs(args[6],sentinel);self.assertTrue(kw['eligible'])
  args,_=eval(compile(ast.Expression(calls['finish']),'actual-transition','eval'),{},env);self.assertIs(args[2],sentinel);self.assertTrue(args[3])
  script="""const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),B=require(process.argv[1]+'/dist/sim/battle.js').Battle,P=require(process.argv[1]+'/dist/sim/pokemon.js').Pokemon,out={};
for(const n of ['tailglow','agility','dragondance']){const lines=[],p={hp:100,isActive:true,boosts:{atk:0,def:0,spa:0,spd:0,spe:0,accuracy:0,evasion:0},battle:{clampIntRange:(v,a,z)=>Math.max(a,Math.min(z,v))},getCappedBoost:P.prototype.getCappedBoost,boostBy:P.prototype.boostBy,toString:()=> 'p2a: Actor'};const m=d.moves.get(n);B.prototype.boost.call({gen:3,runEvent:(id,t,s,e,b)=>b,add:(...v)=>lines.push(v.map(String).join('|'))},m.boosts,p,p,m);out[n]=lines;}process.stdout.write(JSON.stringify(out));"""
  lines=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(lines,{'tailglow':['-boost|p2a: Actor|spa|2'],'agility':['-boost|p2a: Actor|spe|2'],'dragondance':['-boost|p2a: Actor|atk|1','-boost|p2a: Actor|spe|1']})
 def test_caps_called_stale_malformed_and_same_write(self):
  x=self.component();t=x.tape;c=x.capture;self.assertEqual(cb.captured_matches(t,c).tolist(),[True]*6+[False]*2);self.assertEqual(t.count.tolist(),[2,2,3,2,2,3,1,1]);self.assertTrue(bool(j.all((t.refusal[6:]&256)!=0)))
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   bad=t._replace(**{name:getattr(t,name).at[0].add(j.uint32(1))});self.assertFalse(bool(cb.captured_matches(bad,c).any()),name)
   with self.assertRaises(ValueError):cb.captured_matches(t._replace(**{name:j.broadcast_to(getattr(t,name),(8,8))}),c)
  for name in ('turn','phase','generation'):
   bad=t._replace(cursor=t.cursor._replace(**{name:getattr(t.cursor,name)+1}));self.assertFalse(bool(cb.captured_matches(bad,c).any()),name)
  bad=c._replace(new=c.new.at[0,1].add(1));self.assertFalse(bool(cb.captured_matches(t,bad)[0]));self.assertTrue(bool(cb.captured_matches(t,bad)[1]))
  bad=t._replace(words=t.words.at[0,1,4].set(0));self.assertFalse(bool(cb.owned(bad,1)[0]));self.assertTrue(bool(cb.owned(bad,1)[1]))
  sticky=x.before._replace(refusal=x.before.refusal|256)
  sticky,p=cb.begin(sticky,x.state,x.actor,x.move,x.move,x.old,self.runtime['direct_hit_context'],self.table.effects,eligible=j.ones(8,bool));sticky,sc=cb.finish(sticky,p,x.new,j.ones(8,bool));self.assertTrue(bool(j.all((sticky.refusal&256)!=0)))
  off,p=cb.begin(x.before,x.state,x.actor,x.move,x.move,x.old,self.runtime['direct_hit_context'],self.table.effects,eligible=j.zeros(8,bool));off,_=cb.finish(off,p,x.new,j.ones(8,bool));np.testing.assert_array_equal(off.count,x.before.count)
  no_write,p=cb.begin(x.before,x.state,x.actor,x.move,x.move,x.old,self.runtime['direct_hit_context'],self.table.effects,eligible=j.ones(8,bool));no_write,nc=cb.finish(no_write,p,x.old,j.ones(8,bool));self.assertFalse(bool(cb.captured_matches(no_write,nc).any()))
 def test_mixed_original_both_full_words_and_claims(self):
  x=self.component();b=8;z=j.zeros(b,j.int32);yes=j.ones(b,bool)
  frame=life.LifecycleResult(x.state,x.initial.cursor,x.initial,life.snapshot(x.cursor),life.snapshot(x.initial.cursor),~yes,z)
  initial=[]
  for owner in (0,1):
   stream=full.register(x.cursor.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=x.cursor.stream_id,owner_sides=z+owner);r=self.fold(stream,frame,pub);self.assertEqual(r.projection.error.tolist(),[0]*b);initial.append((r.projection.state,r.public))
  tape=x.tape;raw_ok=witness._raw_complete_words(tape,self.runtime['frame_witness_compiled'].device)
  for row in range(6):self.assertTrue(bool(j.all(raw_ok[row,:int(tape.count[row])])),(row,raw_ok[row].tolist()))
  frame=life.LifecycleResult(x.state,tape.cursor,tape,life.snapshot(x.initial.cursor),life.snapshot(tape.cursor),~yes,z);compact=[]
  for owner,(stream,pub)in enumerate(initial):
   closed,r=self.consume(tape.execution.ledger,frame,stream,pub)
   Path('/tmp/pokezero-self-boost-consumed-'+str(owner)+'.json').write_text(json.dumps(dict(stage=closed.stage.tolist(),refusal=closed.refusal.tolist(),words=r.projection.frames.words.tolist(),valid=r.projection.frames.valid.tolist(),error=r.projection.error.tolist()))+'\n')
   self.assertTrue(bool(j.all(closed.stage[:6,0]==3)),closed.refusal.tolist())
   for row in range(6):
    names={side:self.catalog.species.names[side+1]for side in (0,1)};ids={side:f'p{side+1}a: '+names[side]for side in (0,1)};who=row%2;move_name=self.names[row%3];move_name={0:'tailglow',1:'agility',2:'dragondance'}[row%3]
    wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ids[side]}|{names[side]}, L80, M|100/100'for side in (0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog))
    suffix={'tailglow':[('spa',2)],'agility':[('spe',2)],'dragondance':[('atk',1),('spe',1)]}[move_name];lines=[f'|move|{ids[who]}|{self.dex.move_info(move_name).name}|{ids[who]}']+[f'|-boost|{ids[who]}|{stat}|{delta}'for stat,delta in suffix]
    _,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));want=np.asarray([e.words()for p in packets for e in p.events],np.int32);actual=np.asarray(r.projection.frames.words[:,row])[np.asarray(r.projection.frames.valid[:,row])];compact.append(dict(owner=owner,row=row,lines=lines,actual=actual.tolist(),expected=want.tolist(),error=int(r.projection.error[row])));Path('/tmp/pokezero-self-boost-both-full-compact.json').write_text(json.dumps(compact,sort_keys=True)+'\n')
    self.assertEqual(int(r.projection.error[row]),0);np.testing.assert_array_equal(actual,want)
  Path('/tmp/pokezero-self-boost-both-full-compact.json').write_text(json.dumps(compact,sort_keys=True)+'\n')
if __name__=='__main__':unittest.main()
