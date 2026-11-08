"""Literal Thunder Wave/Substitute source transactions; no engine turns."""
import ast,json,subprocess,unittest
from copy import deepcopy
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_thunder_wave_callbacks as cb,engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers
from gpu_engine import transition as tr,public_belief,v4_encoder as v4,public_wire,mechanics as m
from gpu_engine import engine_source_frame_witness as witness,engine_toxic_callbacks as primary
from gpu_engine import engine_substitute_callbacks as sub
from _showdown_root import showdown_root
import test_gpu_engine_hidden_power_substitute_input as fixtures
BASE='6103ebdf195da8f61b4bcb0329386417b78a6207'

class ThunderWaveSubstituteTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.HiddenPowerSubstituteInputTests.setUpClass();f=fixtures.HiddenPowerSubstituteInputTests;cls.f=f
  cls.metadata=deepcopy(f.metadata);dex=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves)
  info=dex.move_info('thunderwave');moves['thunderwave']=info;cls.names=('thunderwave','substitute','surf','sleeptalk')
  flags=json.loads(subprocess.check_output(['node','-e',"console.log(JSON.stringify(require(process.argv[1]).Dex.mod('gen3').moves.get('thunderwave').flags))",str(showdown_root()/'dist/sim/dex.js')],text=True))
  if 'thunderwave' not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append('thunderwave');cls.metadata['moves'].append(None)
  cls.move_id=cls.metadata['vocabulary']['moves'].index('thunderwave')
  cls.metadata['moves'][cls.move_id]=dict(id='thunderwave',base_power=0,type=cls.metadata['vocabulary']['types'].index('electric'),source_type='Electric',category='Status',pp=info.pp,accuracy=100,priority=info.priority,flags=flags)
  cls.dex=ShowdownDex(moves,f.dex.species,f.dex.type_chart)
  variants={name:Gen3RandbatSpeciesUniverse(name,80,(Gen3RandbatVariant('twave-'+name,'authored',name,'authored',80,cls.names,'Pressure','Leftovers'),))for name in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-original-thunder-sub-failure'),variants)
  labels=set(f.catalog.vocabulary.tokens)|{'move:thunderwave','belief:possible_move:thunderwave'}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=f.catalog.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  cls.rules=request.load_source_rules(showdown_root(),cls.metadata);req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer);facts=own.compile_own_facts(req,cls.catalog)
  hit=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.thunder=cb.compile_thunder_wave(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.legacy=cb.compile_thunder_wave(engine_catalog=cls.producer,metadata=cls.metadata)
  substitute=sub.compile_substitute(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.prepared=driver.prepare(req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=hit,direct_hit_mode='kernel-event',thunder_wave_compiled=cls.thunder,thunder_wave_mode='kernel-event',substitute_compiled=substitute,substitute_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))
 component=fixtures.HiddenPowerSubstituteInputTests.component

 def literal(self,consume_initial=True,mask=None):
  s,t,consumers,actor,phys=self.component(consume_initial);s=s._replace(done=j.zeros(8,bool),turn=t.cursor.turn);b=8;rows=j.arange(b);yes=j.ones(b,bool)if mask is None else mask;z=j.zeros(b,j.int32);target=s.active[rows,1-actor];submove=z+self.metadata['vocabulary']['moves'].index('substitute');move=z+self.move_id
  t=public.move(t,1-actor,target,phys,submove,eligible=yes);t,ticket=ex.open_attempt(t,ex.Site.SUBSTITUTE_SELF,1-actor,active=yes);w=sub.Write(z+100,z+75,z+100,z,z,z+25);t=sub.record_device(t,1-actor,target,submove,w,self.runtime['substitute_context'],eligible=yes);t=ex.capture_substitute_write(t,ticket,1-actor,target,submove,w,requested=yes)
  after=s._replace(hp=s.hp.at[rows,1-actor,target].set(75),substitute_hp=s.substitute_hp.at[rows,1-actor,target].set(25))
  t=public.move(t,actor,phys,target,move,eligible=yes);t,ticket=ex.open_attempt(t,ex.Site.PRIMARY_PARALYSIS,1-actor,active=yes)
  attempt=cb.Attempt(yes,yes,~yes,~yes,yes,yes,~yes,~yes,~yes,~yes,yes,z,z,z,z)
  t=cb.record_device(t,ticket,actor,phys,target,move,attempt,self.runtime['thunder_wave_context'],requested=yes)
  return s,after,t,consumers,actor,phys,target,move,attempt,ticket

 def test_original_source_capability_default_and_factory(self):
  script=r'''const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),get=require(root+'/dist/sim/battle-actions.js').BattleActions.prototype.getDamage,lines=[],source={hp:100,status:'',toString:()=> 'p1a: Actor'},target={hp:75,status:'',volatiles:{substitute:{hp:25}},runImmunity:()=>true,toString:()=> 'p2a: Target'},move=d.moves.get('thunderwave'),battle={gen:3,getCategory:m=>m.category};const b={actions:{getDamage:(s,t,m)=>get.call({battle,dex:d},s,t,m)},add:(...xs)=>lines.push(xs.map(String).join('|')),attrLastMove:(...xs)=>lines.push(xs.join('|'))};const result=d.moves.get('substitute').condition.onTryPrimaryHit.call(b,target,source,move);console.log(JSON.stringify({result,lines,hp:target.hp,subhp:target.volatiles.substitute.hp,status:target.status,sourceHP:source.hp}));'''
  got=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(got,dict(result=None,lines=['-fail|p1a: Actor','[still]'],hp=75,subhp=25,status='',sourceHP=100))
  self.assertTrue(bool(self.thunder.device.substitute_failure));self.assertFalse(bool(self.legacy.device.substitute_failure))
  self.assertEqual(Path('gpu_engine/transition.py').read_bytes(),subprocess.check_output(['git','show','b8092a85:gpu_engine/transition.py']))
  s=Path('src/pokezero/v5_worker.py').read_text();self.assertIn("if name in ('recoil','thunder_wave') else {}",s)
  witness._source()

 def test_mixed_current_span_namespaces_and_unsupported(self):
  mask=j.array([True]*6+[False]*2);s,after,t,_,actor,phys,target,move,a,ticket=self.literal(False,mask)
  self.assertEqual(t.count.tolist(),[4]*6+[0,0]);self.assertEqual(cb.span_matches(t,ticket,actor,phys,target,move,a,self.runtime['thunder_wave_context']).tolist(),mask.tolist())
  self.assertEqual(cb.owned_substitute_failure(t,3).tolist(),mask.tolist())
  captured=risk.CapturedThunderWave(self.runtime['thunder_wave_context'],ticket,actor,phys,target,move,a)
  def matches(tape,attempt=a):return risk._captured_thunder_wave_matches(tape,self.runtime['kernel_risk_compiled'].device,captured._replace(attempt=attempt))
  self.assertEqual(matches(t).tolist(),mask.tolist())
  for field,value in (('accuracy_hit',j.zeros(8,bool)),('applied',j.ones(8,bool)),('protected',j.ones(8,bool)),('type_immune',j.ones(8,bool)),('ability_immune',j.ones(8,bool)),('absorbed',j.ones(8,bool)),('damage_immune',j.ones(8,bool)),('old_status',j.ones(8,j.int32)),('old_counter',j.ones(8,j.int32))):self.assertFalse(bool(matches(t,a._replace(**{field:value})).any()),field)
  mutations=[t._replace(words=t.words.at[0,2,5].set(0)),t._replace(event_turn=t.event_turn.at[0,3].add(1)),t._replace(cursor=t.cursor._replace(generation=t.cursor.generation.at[0,1].add(1))),t._replace(cursor=t.cursor._replace(physical_active=t.cursor.physical_active.at[0,0].set(1))),t._replace(execution=t.execution._replace(ledger=t.execution.ledger._replace(claimed=t.execution.ledger.claimed.at[0,3].set(0))))]
  for bad in mutations:self.assertEqual(matches(bad).tolist(),[False]+mask.tolist()[1:])
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   for value in (j.broadcast_to(getattr(t,name),(8,8)),getattr(t,name).astype(j.int32)):
    with self.assertRaises(ValueError):cb.owned_substitute_failure(t._replace(**{name:value}),3)
  raw=primary.words_device(actor,phys,target,move,a);rt=self.runtime['kernel_risk_compiled'].device
  closed=t._replace(execution=t.execution._replace(risks=t.execution.risks._replace(tables=rt._replace(thunder_wave_substitute_failure=j.array(False)))))
  self.assertFalse(bool(cb.substitute_failure_source(closed,raw).any()))
  prior=t._replace(refusal=t.refusal|128);kept=risk.primary_status_on_tape(prior,mask,active=mask,captured_thunder_wave=captured);self.assertTrue(bool((kept.refusal&128).all()))
  trailing=public.move(t,actor,phys,target,move,eligible=mask);self.assertFalse(bool(matches(trailing).any()))

 def test_original_both_full_same_call_substitute_failure(self):
  s,after,t,consumers,actor,phys,target,move,a,ticket=self.literal();yes=j.ones(8,bool);z=j.zeros(8,j.int32)
  capture=risk.CapturedThunderWave(self.runtime['thunder_wave_context'],ticket,actor,phys,target,move,a)
  t=risk.primary_status_on_tape(t,yes,active=yes,captured_thunder_wave=capture);self.assertEqual(t.execution.risks.carry.risk.tolist(),[0]*8);self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))
  t=life.action_end(t,yes);t=life.finish(t,after,live=yes,replacement=~yes,pending_before=~yes,residual_ready=yes,defer_residual=~yes,turn_context=yes)
  self.assertEqual(t.cursor.turn.tolist(),[2]*8);self.assertTrue(bool(cb.owned_substitute_failure(t,3).all()))
  self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))
  frame=life.LifecycleResult(after,t.cursor,t,life.snapshot(t.cursor),life.snapshot(t.cursor),~yes,z);packets=[]
  for owner,(stream,pub)in enumerate(consumers):
   got=self.fold(stream,frame,pub);packets.append(dict(owner=owner,error=got.projection.error.tolist(),admitted=got.projection.component_admitted.tolist(),words=got.projection.frames.words.tolist(),valid=got.projection.frames.valid.tolist()));Path('/tmp/pokezero-thunder-wave-substitute-full-current.json').write_text(json.dumps(packets)+'\n');self.assertEqual(got.projection.error.tolist(),[0]*8);self.assertEqual(got.projection.component_admitted.tolist(),[True]*8)
   for row in range(8):
    names={side:self.catalog.species.names[side+1]for side in(0,1)};ids={side:f'p{side+1}a: '+names[side]for side in(0,1)};who=int(actor[row]);foe=1-who
    wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ids[side]}|{names[side]}, L80, M|100/100'for side in(0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog));_,original=public_wire.project_lines(wire,[f'|move|{ids[foe]}|Substitute|{ids[foe]}',f'|-start|{ids[foe]}|Substitute',f'|-damage|{ids[foe]}|75/100',f'|move|{ids[who]}|Thunder Wave||[still]',f'|-fail|{ids[who]}','|','|upkeep','|turn|2'],**v4.wire_kwargs(self.catalog));want=np.asarray([event.words()for packet in original for event in packet.events],np.int32);actual=np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,want)


class ThunderWaveStaticTests(unittest.TestCase):
 def test_original_factory_same_rules_and_cold_import_orders(self):
  import itertools,sys
  from types import SimpleNamespace
  source=ast.parse(Path('src/pokezero/v5_worker.py').read_text());before=ast.parse(subprocess.check_output(['git','show',BASE+':src/pokezero/v5_worker.py'],text=True))
  current=next(n for n in source.body if isinstance(n,ast.FunctionDef)and n.name=='_prepare_runtime')
  old=next(n for n in before.body if isinstance(n,ast.FunctionDef)and n.name=='_prepare_runtime')
  normalized=ast.parse(ast.unparse(current).replace("name in ('recoil', 'thunder_wave')","name == 'recoil'"))
  self.assertEqual(ast.dump(normalized.body[0],include_attributes=False),ast.dump(old,include_attributes=False))
  loop=next(n for n in ast.walk(current)if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple)and [x.id for x in n.target.elts]==['name','move','module','function'])
  assignment=loop.body[0];code=compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),'<original-worker-compiler-call>','exec')
  rules=object();catalog=object();metadata={'vocabulary':{'moves':['doubleedge','thunderwave','calmmind']}};calls=[]
  for name,move in [('recoil','doubleedge'),('thunder_wave','thunderwave'),('calm_mind','calmmind')]:
   def compiler(**kw):calls.append((name,kw));return 'actual-compiler-result'
   env=dict(name=name,move=move,module=SimpleNamespace(compile=compiler),function='compile',lifecycle=catalog,metadata=metadata,rules=rules)
   exec(code,env);self.assertEqual(env['helper'],'actual-compiler-result')
  self.assertEqual([name for name,kw in calls if kw.get('source_rules')is rules],['recoil','thunder_wave'])
  for name,kw in calls:self.assertIs(kw['engine_catalog'],catalog);self.assertIs(kw['metadata'],metadata)
  modules=('gpu_engine.benchmark_retained_games','gpu_engine.engine_lifecycle_callbacks','gpu_engine.engine_full_v4_event_device')
  for order in itertools.permutations(modules):subprocess.run([sys.executable,'-B','-c',';'.join('import '+name for name in order)],check=True,capture_output=True,text=True,timeout=30)

if __name__=='__main__':unittest.main()
