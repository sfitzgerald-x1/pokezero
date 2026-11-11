"""Original accuracy MISS before Protect, no numeric change."""
import ast,json,subprocess,unittest
from copy import deepcopy
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_protect_callbacks as cb,engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers
from gpu_engine import transition as tr,public_belief,v4_encoder as v4,public_wire,state
from gpu_engine import engine_source_frame_witness as witness
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_no_damage_owning as fixtures
from gpu_engine import engine_direct_ko_callbacks as ko
from types import SimpleNamespace
from gpu_engine import engine_no_damage_callbacks as nd,absorption
BASE='ad123786c29ef068b53ca6f707db0313098856a6'
class ProtectAccuracyMissInputTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.NoDamageOwningTests.setUpClass();f=fixtures.NoDamageOwningTests;cls.f=f;cls.metadata=deepcopy(f.metadata)
  dex=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves);cls.names=('return','protect','surf','sleeptalk')
  for name in cls.names:
   info=dex.move_info(name);moves[name]=info
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   i=cls.metadata['vocabulary']['moves'].index(name);cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=True if name in ('protect','sleeptalk')else 100,priority=info.priority,flags={}if name!='surf'else{'protect':1})
  # This shared catalog also contains Encore; keep that unused fixture row
  # equal to the original installed move rather than the generic always-hit row.
  if 'encore' in cls.metadata['vocabulary']['moves']:
   encore=dex.move_info('encore');moves['encore']=encore
   cls.metadata['moves'][cls.metadata['vocabulary']['moves'].index('encore')]=dict(id='encore',base_power=encore.base_power,type=cls.metadata['vocabulary']['types'].index(encore.type.lower()),source_type=encore.type,category=encore.category,pp=encore.pp,accuracy=100,priority=encore.priority,flags={'protect':1,'mirror':1,'bypasssub':1,'metronome':1,'failencore':1})
  cls.dex=ShowdownDex(moves,f.dex.species,f.dex.type_chart)
  variants={name:Gen3RandbatSpeciesUniverse(name,80,(Gen3RandbatVariant('heal-'+name,'authored',name,'authored',80,cls.names,'Pressure','Leftovers'),))for name in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-original-self-heal'),variants)
  labels=set(f.catalog.vocabulary.tokens)
  for n in cls.names:labels|={'move:'+n,'belief:possible_move:'+n}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=f.catalog.masks.__dict__)
  # Keep the authored installed flags equal to the pinned original source.
  original_flags=json.loads(subprocess.check_output(['node','-e',"const d=require(process.argv[1]).Dex.mod('gen3');console.log(JSON.stringify(JSON.parse(process.argv[2]).map(n=>!!d.moves.get(n).flags.failencore)))",str(showdown_root()/'dist/sim/dex.js'),json.dumps(cls.metadata['vocabulary']['moves'])],text=True))
  for row,flag in zip(cls.metadata['moves'],original_flags):
   if row is not None:row.setdefault('flags',{})['failencore']=flag
  for n in cls.names:
   flags=json.loads(subprocess.check_output(['node','-e',"const d=require(process.argv[1]).Dex.mod('gen3');console.log(JSON.stringify(d.moves.get(process.argv[2]).flags))",str(showdown_root()/'dist/sim/dex.js'),n],text=True));cls.metadata['moves'][cls.metadata['vocabulary']['moves'].index(n)]['flags']=flags
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  rules=request.load_source_rules(showdown_root(),cls.metadata);cls.rules=rules;req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=rules,lifecycle_catalog=cls.producer);facts=own.compile_own_facts(req,cls.catalog)
  cls.rules=rules;cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=rules)
  cls.prepared=driver.prepare(req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))

 def component(self,consume_initial=True):
  b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);actor=rows%2;active=j.stack((z+3,z+4),axis=1);phys=active[rows,actor]
  s=state.empty_state(b)._replace(active=active,gender=j.ones((b,2,6),j.int32),hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),level=j.full((b,2,6),80,j.int32))
  kw=dict(catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32));cursor=life.register(rows+97001,**kw);initial=life._tape(cursor,self.producer,64)
  for side in(0,1):initial=life._reveal(initial,s,z+side,z-1,active[:,side],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes);frame=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(cursor),life.snapshot(initial.cursor),~yes,z);consumers=[]
  for owner in((0,1)if consume_initial else()):
   stream=full.register(cursor.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=cursor.stream_id,owner_sides=z+owner);r=self.fold(stream,frame,pub);self.assertEqual(r.projection.error.tolist(),[0]*b);consumers.append((r.projection.state,r.public))
  tape=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device;tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,rt),tape,rt,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,rt)))
  return s,tape,consumers,actor,phys


 def make_miss(self,consume_initial=False):
  s,t,consumers,actor,phys=self.component(consume_initial);b=8;rows=j.arange(b);yes=j.ones(b,bool);no=~yes;z=j.zeros(b,j.int32);foe=1-actor;target=s.active[rows,foe];ctx=self.runtime['direct_hit_context'];move=z+self.metadata['vocabulary']['moves'].index('hydropump');protect=z+self.metadata['vocabulary']['moves'].index('protect')
  t=public.move(t,foe,target,phys,protect,eligible=yes);t,pp=cb.begin(t,s,foe,protect,z,z,yes,yes,yes,ctx,self.table.effects,eligible=yes);s=s._replace(protect_counter=s.protect_counter.at[rows,foe,target].set(2));t=cb.finish(t,pp,s,yes,yes,yes)
  t=public.move(t,actor,phys,target,move,eligible=yes);t,bp=cb.begin_block(t,s,actor,move,no,yes,yes,ctx,eligible=yes);t,bc=cb.finish_block(t,bp,no)
  self.assertEqual(t.execution.ledger.count.tolist(),[1]*8);self.assertEqual(t.execution.ledger.refusal.tolist(),[0]*8)
  t,ticket=ex.open_attempt(t,ex.Site.DIRECT_HIT,foe,active=yes);at=(rows,foe,target);before=nd.Before(no,z+self.metadata['vocabulary']['types'].index('water'),s.ability[at],s.status[at],s.hp[at],s.max_hp[at],s.volatile[at],s.substitute_hp[at],s.hp[rows,actor,phys],no,s.hidden_power_type[rows,actor,phys],s.hidden_power_power[rows,actor,phys])
  p=nd.prepare(t,ticket,actor,phys,target,move,before,ctx,requested=yes,caller_move=move);w=direct.HitWrite(z+120,no,no,no,z+1,z+1,no,yes,no,before.hp,before.hp,before.maximum);o=risk.Outcome(w.power,w.critical,w.hit,w.immune,w.numerator,w.denominator,no,w.substitute,w.protected,w.enduring,z,no)
  return s,t,consumers,p,w,o

 def test_original_order_default_and_refusals(self):
  script=r"""const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),f=require(root+'/dist/data/mods/gen3/scripts.js').Scripts.actions.tryMoveHit,out=[];for(const passes of[false,true]){let calls=0,lines=[],move={...d.moves.get('hydropump')},source={boosts:{accuracy:0},getVolatile:()=>null,toString:()=> 'p1a: Actor'},target={boosts:{evasion:0},runImmunity:()=>true,toString:()=> 'p2a: Target'},b={setActiveMove(){},singleEvent:()=>true,runEvent(name,t,s,e,n){if(name==='TryHit'){calls++;return d.conditions.get('protect').onTryHit.call(b,t,s,e)}if(['ModifyBoost','ModifyAccuracy','Accuracy'].includes(name))return n;return true},clampIntRange:x=>x,randomChance:()=>passes,checkMoveBypassesProtect:()=>false,attrLastMove:(...x)=>lines.push(x.join('|')),add:(...x)=>lines.push(x.map(String).join('|'))};out.push({passes,result:f.call({battle:b},target,source,move),calls,lines});}console.log(JSON.stringify(out));"""
  original=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual([x['calls']for x in original],[0,1]);self.assertEqual(original[0]['lines'],['[miss]','-miss|p1a: Actor|p2a: Target']);self.assertEqual(original[1]['lines'],['-activate|p2a: Target|Protect'])
  s,t,_,p,w,o=self.make_miss();self.assertEqual(nd._code(p,w).tolist(),[nd.MISS]*8)
  # Original validator refuses the SAME actual Before.accuracy_hit=False,
  # final protected=True tuple; old_hit is not reconstructed from R4.
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_no_damage_callbacks.py'],text=True));fn=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name=='_code');scope=dict(nd.__dict__);exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original no-damage>','exec'),scope);self.assertEqual(scope['_code'](p,w).tolist(),[0]*8)
  self.assertFalse(bool(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.no_damage.protect_miss))
  z=j.zeros(8,j.int32);yes=j.ones(8,bool)
  for pp,ww in[(p._replace(caller_move=None),w),(p._replace(caller_move=z+self.metadata['vocabulary']['moves'].index('sleeptalk')),w),(p._replace(before=p.before._replace(accuracy_hit=yes)),w),(p._replace(before=p.before._replace(sub_hp=z+25)),w),(p,w._replace(enduring=yes)),(p,w._replace(immune=yes,numerator=z)),(p,w._replace(new_hp=z+99)),(p._replace(absorbed=p.absorbed._replace(blocked=yes)),w),(p._replace(context=p.context._replace(data=p.context.data._replace(protect=None))),w)]:self.assertFalse(bool(nd.scheduled(pp,ww).any()))
  with self.assertRaises(ValueError):nd.scheduled(p._replace(caller_move=j.broadcast_to(p.move,(8,8))),w)
  oldtree=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));new=ast.parse(subprocess.check_output(['git','show','a8034bc2:gpu_engine/transition.py'],text=True));baseline=dump_default_program(oldtree);self.assertEqual(dump_default_program(new),baseline);assert_hp_rng_mutations_detected(self,new,baseline)
  current=ast.parse(Path('gpu_engine/transition.py').read_text());integrated=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/transition.py'],text=True))
  self.assertEqual(ast.dump(current),ast.dump(integrated));assert_hp_rng_mutations_detected(self,current,dump_default_program(integrated))
  # Existing pre-Protect accuracy capture is before numeric Protect hit &=.
  text=Path('gpu_engine/transition.py').read_text();self.assertLess(text.index('_direct_accuracy_hit=hit'),text.index('protect_callbacks.begin_block'));self.assertLess(text.index('protect_callbacks.begin_block'),text.index('hit &= ~(protected'))
  witness._source()

 def test_typed_protect_then_accuracy_miss_both_full(self):
  s,t,consumers,p,w,o=self.make_miss(True);rows=j.arange(8);yes=j.ones(8,bool);z=j.zeros(8,j.int32);before=t
  t=nd.finish(t,p,w,o,p.before.actor_hp,p.before.volatile,p.before.sub_hp,p.before.actor_hp_type,p.before.actor_hp_power);c=nd.Captured(p,w,p.before.actor_hp,p.before.volatile,p.before.sub_hp,p.before.actor_hp_type,p.before.actor_hp_power)
  self.assertTrue(bool(nd.captured_matches(t,c,o).all()));self.assertEqual(t.count.tolist(),[5]*8);self.assertEqual(t.words[:,3,6].tolist(),[nd.MISS]*8);self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))
  r=risk.outcome_on_tape(t,o,eligible=yes,active=yes,captured_no_damage=c);self.assertEqual(r.execution.risks.carry.risk.tolist(),[0]*8)
  frame=life.LifecycleResult(s,t.cursor,t,life.snapshot(t.cursor),life.snapshot(t.cursor),~yes,z);records=[]
  for owner,(stream,pub)in enumerate(consumers):
   got=self.fold(stream,frame,pub);records.append(dict(owner=owner,error=got.projection.error.tolist(),words=got.projection.frames.words.tolist(),valid=got.projection.frames.valid.tolist()));Path('/tmp/pokezero-protect-accuracy-miss-full-current.json').write_text(json.dumps(records)+'\n');self.assertEqual(got.projection.error.tolist(),[0]*8)
   for row in range(8):
    who=int(p.actor[row]);foe=1-who;ids={side:f'p{side+1}a: '+self.catalog.species.names[side+1]for side in(0,1)};initial=[f'|switch|{ids[side]}|{self.catalog.species.names[side+1]}, L80, M|100/100'for side in(0,1)];wire,_=public_wire.project_lines(public_wire.WireState(),initial+['|turn|1'],**v4.wire_kwargs(self.catalog));lines=[f'|move|{ids[foe]}|Protect|{ids[foe]}',f'|-singleturn|{ids[foe]}|Protect',f'|move|{ids[who]}|Hydro Pump|{ids[foe]}|[miss]',f'|-miss|{ids[who]}|{ids[foe]}'];_,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));expected=np.asarray([e.words()for p in packets for e in p.events],np.int32);actual=np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,expected)
  self.assertFalse(bool(nd.captured_matches(t._replace(event_turn=t.event_turn.at[:,2].add(1)),c,o).any()))
  old=t._replace(refusal=t.refusal|128);r=risk.outcome_on_tape(old,o,eligible=yes,active=yes,captured_no_damage=c);self.assertTrue(bool((r.refusal&128).all()))
  no_protect=w._replace(protected=~yes);self.assertEqual(nd._code(p._replace(caller_move=None),no_protect).tolist(),[nd.MISS]*8)

if __name__=='__main__':unittest.main()
