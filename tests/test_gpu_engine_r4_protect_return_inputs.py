"""Source-only original self heal callback and BOTH FULL components."""
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
BASE='ac13fdbd88d0653444288f2d87927624181c946c'
class R4ProtectReturnInputTests(unittest.TestCase):
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
  cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=rules)
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

 def test_original_accuracy_tryhit_and_false_obligation(self):
  script=r'''const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),f=require(root+'/dist/data/mods/gen3/scripts.js').Scripts.actions.tryMoveHit,out=[];
for(const passes of [false,true]){let calls=0,lines=[],m={...d.moves.get('hydropump')},actor={level:80,boosts:{accuracy:0},toString:()=> 'p1a: Actor'},target={level:80,boosts:{evasion:0},runImmunity:()=>true,toString:()=> 'p2a: Target'};const b={setActiveMove(){},singleEvent:()=>true,runEvent(name,t,s,e,n){if(name==='TryHit'){calls++;return null}if(name==='ModifyBoost')return n;if(name==='ModifyAccuracy'||name==='Accuracy')return n;return true},clampIntRange:x=>x,randomChance:()=>passes,attrLastMove:(...x)=>lines.push(x.join('|')),add:(...x)=>lines.push(x.map(String).join('|'))};out.push({passes,result:f.call({battle:b},target,actor,m),tryHit:calls,lines})}console.log(JSON.stringify(out));'''
  got=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual([x['tryHit']for x in got],[0,1]);self.assertEqual(got[0]['lines'],['[miss]','-miss|p1a: Actor|p2a: Target'])
  s,t,_,actor,phys=self.component(consume_initial=False);b=8;yes=j.ones(b,bool);no=~yes;move=j.zeros(b,j.int32)+self.metadata['vocabulary']['moves'].index('surf')
  t=public.move(t,actor,phys,s.active[j.arange(b),1-actor],move,eligible=yes)
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_protect_callbacks.py'],text=True));func=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name=='begin_block');scope=dict(cb.__dict__);exec(compile(ast.Module(body=[func],type_ignores=[]),'<original false obligation>','exec'),scope)
  before,p=scope['begin_block'](t,s,actor,move,no,yes,yes,self.runtime['direct_hit_context'],eligible=yes);before,_=cb.finish_block(before,p,no);self.assertEqual(before.execution.ledger.count.tolist(),[1]*b);self.assertTrue(bool((before.execution.ledger.refusal!=0).all()))
  after,p=cb.begin_block(t,s,actor,move,no,yes,yes,self.runtime['direct_hit_context'],eligible=yes);after,_=cb.finish_block(after,p,no);self.assertEqual(after.execution.ledger.count.tolist(),[0]*b);self.assertEqual(after.execution.ledger.refusal.tolist(),[0]*b);np.testing.assert_array_equal(after.words,t.words)
  # Genuine accuracy hit still requests an obligation and missing Protect source refuses.
  bad,p=cb.begin_block(t,s,actor,move,yes,yes,yes,self.runtime['direct_hit_context'],eligible=yes);bad,_=cb.finish_block(bad,p,no);self.assertTrue(bool((bad.execution.ledger.refusal!=0).all()))

 def test_return_same_write_happiness_ko_both_full_and_unknowns(self):
  witness._source();s,start,consumers,actor,phys=self.component();b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);move=z+self.metadata['vocabulary']['moves'].index('return');target=s.active[rows,1-actor];at=(rows,1-actor,target)
  self.assertEqual(int(self.helper.device.return_ko_move),int(move[0]));self.assertEqual(int(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.return_ko_move),0)
  t=public.move(start,actor,phys,target,move,eligible=yes);t,ticket=ex.open_attempt(t,ex.Site.DIRECT_HIT,1-actor,active=yes)
  happiness=j.array([255,128,0,1,255,128,0,1],j.int32);powers=j.maximum(1,happiness*10//25);s=s._replace(happiness=s.happiness.at[rows,actor,phys].set(happiness));w=direct.HitWrite(powers,~yes,yes,~yes,z+1,z+1,~yes,~yes,~yes,z+100,z,z+100)
  t,p=ko.prepare_controls_device(t,ticket,actor,phys,target,move,w,self.runtime['direct_hit_context'],z+100,s.hp[rows,1-actor],eligible=yes,actor_happiness=s.happiness[rows,actor,phys]);self.assertTrue(bool(ko._literal(p,b).all()))
  for bad in (p._replace(actor_happiness=None),p._replace(actor_happiness=j.full(b,256,j.int32)),p._replace(write=w._replace(power=powers+1))):self.assertFalse(bool(ko._literal(bad,b).any()))
  after=s._replace(hp=s.hp.at[at].set(0));t=public.condition(t,1-actor,target,z,z+100,z,eligible=yes)
  o=risk.Outcome(powers,~yes,yes,~yes,z+1,z+1,~yes,~yes,~yes,~yes,z,~yes)
  self.assertTrue(bool(ko.matches_outcome_device(t,p,o,self.runtime['kernel_risk_compiled'].device).all()))
  t=public.faint(t,1-actor,target,eligible=yes);t=ko.finish_device(t,p,z+100,after.hp[rows,1-actor],phys,target);self.assertEqual(t.refusal.tolist(),[1]*b)
  self.assertEqual(t.execution.ledger.stage[:,0].tolist(),[2]*b)
  frame=life.LifecycleResult(after,t.cursor,t,life.snapshot(start.cursor),life.snapshot(t.cursor),~yes,z);records=[]
  for owner,(stream,pub)in enumerate(consumers):
   got=self.fold(stream,frame,pub);records.append(dict(owner=owner,error=got.projection.error.tolist(),words=got.projection.frames.words.tolist(),valid=got.projection.frames.valid.tolist()));Path('/tmp/pokezero-r4-return-ko-full-current.json').write_text(json.dumps(records)+'\n');self.assertEqual(got.projection.error.tolist(),[0]*b)
   for row in range(b):
    names={seat:self.catalog.species.names[seat+1]for seat in(0,1)};ids={seat:f'p{seat+1}a: '+names[seat]for seat in(0,1)};who=int(actor[row]);foe=1-who
    wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ids[seat]}|{names[seat]}, L80, M|100/100'for seat in(0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog));_,packets=public_wire.project_lines(wire,[f'|move|{ids[who]}|Return|{ids[foe]}',f'|-damage|{ids[foe]}|0 fnt',f'|faint|{ids[foe]}'],**v4.wire_kwargs(self.catalog));expected=np.asarray([e.words()for packet in packets for e in packet.events],np.int32);actual=np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,expected)
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));new=ast.parse(subprocess.check_output(['git','show','0961c0e5:gpu_engine/transition.py'],text=True));baseline=dump_default_program(old);self.assertEqual(dump_default_program(new),baseline);assert_hp_rng_mutations_detected(self,new,baseline)
  current=ast.parse(Path('gpu_engine/transition.py').read_text());integrated=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/transition.py'],text=True))
  self.assertEqual(ast.dump(current),ast.dump(integrated));assert_hp_rng_mutations_detected(self,current,dump_default_program(integrated))

if __name__=='__main__':unittest.main()
