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
from gpu_engine import engine_toxic_callbacks as cb,engine_direct_hit_controls as direct
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
from gpu_engine import engine_substitute_callbacks as sub
BASE='2dbb27e8f61123ca75a0ef570bb4f9f8eea7fe72'
class ToxicSubstituteFailureTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.NoDamageOwningTests.setUpClass();f=fixtures.NoDamageOwningTests;cls.f=f;cls.metadata=deepcopy(f.metadata)
  dex=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves);cls.names=('toxic','substitute','surf','sleeptalk')
  for name in cls.names:
   info=dex.move_info(name);moves[name]=info
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   i=cls.metadata['vocabulary']['moves'].index(name);cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=85 if name=='toxic'else True if name in ('sleeptalk','substitute')else 100,priority=info.priority,flags={}if name!='surf'else{'protect':1})
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
  cls.toxic=cb.compile_toxic(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=rules);cls.substitute=sub.compile_substitute(engine_catalog=cls.producer,metadata=cls.metadata)
  cls.prepared=driver.prepare(req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',toxic_compiled=cls.toxic,toxic_mode='kernel-event',substitute_compiled=cls.substitute,substitute_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
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

 def test_original_effective_callback_and_closed_variants(self):
  script=r'''const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),get=require(root+'/dist/sim/battle-actions.js').BattleActions.prototype.getDamage,lines=[],source={hp:100,status:'',toString:()=> 'p1a: Actor'},target={hp:75,status:'',volatiles:{substitute:{hp:25}},runImmunity:()=>true,toString:()=> 'p2a: Target'},move=d.moves.get('toxic'),battle={gen:3,getCategory:m=>m.category};const b={actions:{getDamage:(s,t,m)=>get.call({battle,dex:d},s,t,m)},add:(...xs)=>lines.push(xs.map(String).join('|')),attrLastMove:(...xs)=>lines.push(xs.join('|'))};const result=d.moves.get('substitute').condition.onTryPrimaryHit.call(b,target,source,move);console.log(JSON.stringify({result,lines,hp:target.hp,subhp:target.volatiles.substitute.hp,status:target.status,sourceHP:source.hp}));'''
  got=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(got,dict(result=None,lines=['-fail|p1a: Actor','[still]'],hp=75,subhp=25,status='',sourceHP=100))
  self.assertTrue(bool(self.toxic.device.substitute_failure));self.assertFalse(bool(cb.compile_toxic(engine_catalog=self.producer,metadata=self.metadata).device.substitute_failure))
  raw=j.array([[26,1,0,0,3,self.metadata['vocabulary']['moves'].index('toxic'),1075,0,0,0]],j.int32);self.assertTrue(bool(cb.substitute_failure_raw(raw)[0]));self.assertFalse(bool(cb.supported_raw(raw)[0]))
  for flags in (1075|cb.TYPE_IMMUNE,1075|cb.ABILITY_IMMUNE,1075|cb.PROTECTED,1075|cb.ABSORBED,1075|cb.DAMAGE_IMMUNE,1075&~cb.ACCURACY_HIT):self.assertFalse(bool(cb.substitute_failure_raw(raw.at[0,6].set(flags))[0]))
  self.assertFalse(bool(cb.substitute_failure_raw(raw.at[0,7:9].set(4))[0]))

 def literal(self):
  s,t,consumers,actor,phys=self.component();b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);target=s.active[rows,1-actor];submove=z+self.metadata['vocabulary']['moves'].index('substitute');move=z+self.metadata['vocabulary']['moves'].index('toxic')
  t=public.move(t,1-actor,target,phys,submove,eligible=yes);t,ticket=ex.open_attempt(t,ex.Site.SUBSTITUTE_SELF,1-actor,active=yes);w=sub.Write(z+100,z+75,z+100,z,z,z+25);t=sub.record_device(t,1-actor,target,submove,w,self.runtime['substitute_context'],eligible=yes);t=ex.capture_substitute_write(t,ticket,1-actor,target,submove,w,requested=yes)
  after=s._replace(hp=s.hp.at[rows,1-actor,target].set(75),substitute_hp=s.substitute_hp.at[rows,1-actor,target].set(25))
  t=public.move(t,actor,phys,target,move,eligible=yes);t,ticket=ex.open_attempt(t,ex.Site.TOXIC_ATTEMPT,1-actor,active=yes)
  attempt=cb.Attempt(yes,yes,~yes,~yes,after.substitute_hp[rows,1-actor,target]>0,yes,~yes,~yes,~yes,~yes,yes,z,z,z,z)
  t=cb.record_device(t,actor,phys,target,move,attempt,self.runtime['toxic_context'],eligible=yes);t=ex.capture_toxic_attempt(t,ticket,actor,phys,target,move,attempt,requested=yes)
  return s,after,t,consumers,actor,phys,target,move,attempt,ticket

 def test_same_call_sub_creation_fail_both_full_source_and_mutations(self):
  s,after,t,consumers,actor,phys,target,move,attempt,ticket=self.literal();b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32)
  self.assertEqual(t.words[:,3,6].tolist(),[1075]*b);self.assertTrue(bool(cb.owned_substitute_failure(t,3).all()));self.assertEqual(t.refusal.tolist(),[1]*b)
  captured=risk.CapturedToxic(self.runtime['toxic_context'],ticket,actor,phys,target,move,attempt);rt=self.runtime['kernel_risk_compiled'].device
  t=risk.primary_status_on_tape(t,yes,active=yes,captured=captured);self.assertEqual(t.execution.risks.carry.risk.tolist(),[0]*b);self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))
  frame=life.LifecycleResult(after,t.cursor,t,life.snapshot(t.cursor),life.snapshot(t.cursor),~yes,z);packets=[]
  for owner,(stream,pub)in enumerate(consumers):
   got=self.fold(stream,frame,pub);packets.append(dict(owner=owner,error=got.projection.error.tolist(),words=got.projection.frames.words.tolist(),valid=got.projection.frames.valid.tolist()));Path('/tmp/pokezero-toxic-sub-failure-full-current.json').write_text(json.dumps(packets)+'\n');self.assertEqual(got.projection.error.tolist(),[0]*b)
   for row in range(b):
    names={seat:self.catalog.species.names[seat+1]for seat in(0,1)};ids={seat:f'p{seat+1}a: '+names[seat]for seat in(0,1)};who=int(actor[row]);foe=1-who
    wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ids[seat]}|{names[seat]}, L80, M|100/100'for seat in(0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog));_,original=public_wire.project_lines(wire,[f'|move|{ids[foe]}|Substitute|{ids[foe]}',f'|-start|{ids[foe]}|Substitute',f'|-damage|{ids[foe]}|75/100',f'|move|{ids[who]}|Toxic||[still]',f'|-fail|{ids[who]}'],**v4.wire_kwargs(self.catalog));expected=np.asarray([e.words()for p in original for e in p.events],np.int32);actual=np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,expected)
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   with self.assertRaises(ValueError):cb.owned_substitute_failure(t._replace(**{name:j.broadcast_to(getattr(t,name),(8,8))}),3)
  for bad in (t._replace(words=t.words.at[:,2,5].set(0)),t._replace(event_turn=t.event_turn.at[:,2].add(1)),t._replace(execution=t.execution._replace(ledger=t.execution.ledger._replace(claimed=t.execution.ledger.claimed.at[:,3].set(0))))):self.assertFalse(bool(cb.owned_substitute_failure(bad,3).any()))
  prior=t._replace(refusal=t.refusal|128);kept=risk.primary_status_on_tape(prior,yes,active=yes,captured=captured);self.assertTrue(bool((kept.refusal&128).all()))
  # Source/default numerics and original closed APIs remain exact.
  baseline=dump_default_program(ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True)))
  feature=ast.parse(subprocess.check_output(['git','show','098c8898:gpu_engine/transition.py'],text=True))
  self.assertEqual(dump_default_program(feature),baseline)
  assert_hp_rng_mutations_detected(self,feature,baseline)
  current=ast.parse(Path('gpu_engine/transition.py').read_text())
  integrated=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/transition.py'],text=True))
  self.assertEqual(ast.dump(current),ast.dump(integrated))
  assert_hp_rng_mutations_detected(self,current,dump_default_program(integrated));witness._source()

if __name__=='__main__':unittest.main()
