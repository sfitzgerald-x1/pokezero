"""Source-only original self heal callback and BOTH FULL components."""
import json,subprocess,unittest
from copy import deepcopy
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_self_move_callbacks as cb,engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers
from gpu_engine import transition as tr,public_belief,v4_encoder as v4,public_wire,state
from gpu_engine import engine_source_frame_witness as witness
from _showdown_root import showdown_root
import test_gpu_engine_no_damage_owning as fixtures
BASE='5b702e55a9a68f2ef7141b7a0bad35190861a60b'

class SelfMoveOwningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.NoDamageOwningTests.setUpClass();f=fixtures.NoDamageOwningTests;cls.f=f;cls.metadata=deepcopy(f.metadata)
  dex=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves);cls.names=('rest','sleeptalk','wish','surf')
  for name in cls.names:
   info=dex.move_info(name);moves[name]=info
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   i=cls.metadata['vocabulary']['moves'].index(name);cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=True if name!='surf'else 100,priority=info.priority,flags={}if name!='surf'else{'protect':1})
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
  # Original denied Rest envelopes remain explicit in the success fixture.
  for family,name in (('abilities','insomnia'),('items','lumberry')):
   if name not in cls.metadata['vocabulary'][family]:cls.metadata['vocabulary'][family].append(name)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  rules=request.load_source_rules(showdown_root(),cls.metadata);req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=rules,lifecycle_catalog=cls.producer);facts=own.compile_own_facts(req,cls.catalog)
  cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=rules)
  cls.prepared=driver.prepare(req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))

 def test_source_default_and_cold_imports(self):
  witness._source();self.assertEqual(len(witness.SOURCE_PINS),89)
  self.assertIsNone(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.self_move)
  # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
  for modules in [('benchmark_retained_games','engine_lifecycle_event_device','engine_full_v4_event_device'),('engine_full_v4_event_device','engine_lifecycle_event_device','benchmark_retained_games')]:
   subprocess.run([__import__('sys').executable,'-c',';'.join('import gpu_engine.'+n for n in modules)],check=True,capture_output=True)
  # Actual effective original callbacks, no Battle/Native/game construction.
  script=r'''const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),lines=[],p={hp:100,maxhp:100,status:'',hasAbility:()=>false,toString:()=> 'p1a: Actor'},b={add:(...xs)=>lines.push(xs.map(String).join('|'))};const rest=d.moves.get('rest').onTry.call(b,p),sleep=d.moves.get('sleeptalk').onTry.call(b,p);console.log(JSON.stringify({rest,sleep,lines,wish:[d.moves.get('wish').condition.duration,d.moves.get('wish').condition.onResidualOrder]}));'''
  got=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(got,dict(rest=None,sleep=False,lines=['-fail|p1a: Actor|heal'],wish=[2,7]))

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

 def test_three_literal_paths_both_owners_and_refusal(self):
  s,start,consumers,actor,phys=self.component();b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);names=self.metadata['vocabulary']['moves'];all_words=[]
  for code,name in ((cb.REST_FAIL,'rest'),(cb.SLEEP_FAIL,'sleeptalk'),(cb.WISH_START,'wish')):
   move=z+names.index(name);t=public.move(start,actor,phys,s.active[rows,1-actor],move,eligible=yes);t,p=cb.begin(t,s,actor,move,self.runtime['direct_hit_context'],code=code,eligible=yes,secondary=j.full(b,code!=cb.SLEEP_FAIL))
   after=s if code!=cb.WISH_START else s._replace(wish_counter=s.wish_counter.at[rows,actor].set(2),wish_amount=s.wish_amount.at[rows,actor].set(50))
   t,c=cb.finish(t,p,after);self.assertTrue(bool(cb.captured_matches(t,c).all()))
   if code==cb.WISH_START:
    t=life.action_end(t,yes)
    t,p=cb.begin(t,after,actor,move,self.runtime['direct_hit_context'],code=cb.WISH_TICK,eligible=yes);after=after._replace(wish_counter=after.wish_counter.at[rows,actor].set(1));t,c=cb.finish(t,p,after);self.assertTrue(bool(cb.captured_matches(t,c).all()))
   frame=life.LifecycleResult(after,t.cursor,t,life.snapshot(start.cursor),life.snapshot(t.cursor),~yes,z)
   for owner,(stream,pub)in enumerate(consumers):
    r=self.fold(stream,frame,pub);all_words.append(dict(code=code,owner=owner,error=r.projection.error.tolist(),words=r.projection.frames.words.tolist(),valid=r.projection.frames.valid.tolist()));Path('/tmp/pokezero-self-move-full-literal-current.json').write_text(json.dumps(all_words)+'\n');self.assertEqual(r.projection.error.tolist(),[0]*b)
    for row in range(b):
     ids={side:f'p{side+1}a: '+self.catalog.species.names[side+1]for side in(0,1)};who=int(actor[row]);initial=[f'|switch|{ids[side]}|{self.catalog.species.names[side+1]}, L80, M|100/100'for side in(0,1)]
     wire,_=public_wire.project_lines(public_wire.WireState(),initial+['|turn|1'],**v4.wire_kwargs(self.catalog));lines=[f'|move|{ids[who]}|{self.dex.move_info(name).name}|'+(''if code==cb.SLEEP_FAIL else ids[who])+('|[still]'if code==cb.SLEEP_FAIL else'')]
     if code==cb.REST_FAIL:lines.append(f'|-fail|{ids[who]}|heal')
     if code==cb.SLEEP_FAIL:lines.append(f'|-fail|{ids[who]}')
     if code==cb.WISH_START:lines.append('|')
     _,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));want=np.asarray([e.words()for p in packets for e in p.events],np.int32);actual=np.asarray(r.projection.frames.words[:,row])[np.asarray(r.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,want)
   # Namespace, source prefix, and current-clock mutations remain refused.
   for field in('engine_binding','source_binding','move_binding','producer_binding'):
    with self.assertRaises(ValueError):cb.owned(t._replace(**{field:j.broadcast_to(getattr(t,field),(8,8))}),1)
   self.assertFalse(bool(cb.owned(t._replace(event_turn=t.event_turn.at[:,1].add(1)),1).any()))
   self.assertFalse(bool(cb.captured_matches(t._replace(cursor=t.cursor._replace(turn=t.cursor.turn+1)),c).any()))
  # An unproved Rest afterstate, sleeping Sleep Talk, absent Wish history/expiry stay UNKNOWN.
  for code,name,bad in ((cb.REST_FAIL,'rest',s._replace(hp=s.hp.at[rows,actor,phys].set(99))),(cb.SLEEP_FAIL,'sleeptalk',s._replace(status=s.status.at[rows,actor,phys].set(5))),(cb.WISH_TICK,'wish',s._replace(wish_counter=s.wish_counter.at[rows,actor].set(1)))):
   move=z+names.index(name);t=public.move(start,actor,phys,bad.active[rows,1-actor],move,eligible=yes);t,p=cb.begin(t,bad,actor,move,self.runtime['direct_hit_context'],code=code,eligible=yes);t,c=cb.finish(t,p,bad);self.assertFalse(bool(cb.captured_matches(t,c).any()));self.assertTrue(bool((t.refusal!=0).all()))

 def test_literal_dispatch_missing_history_and_cross_variant_refusals(self):
  s,start,_,actor,phys=self.component(consume_initial=False);b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);names=self.metadata['vocabulary']['moves'];tables=self.runtime['frame_witness_compiled'].device
  for code,name in ((cb.REST_FAIL,'rest'),(cb.SLEEP_FAIL,'sleeptalk'),(cb.WISH_START,'wish')):
   move=z+names.index(name);t=public.move(start,actor,phys,s.active[rows,1-actor],move,eligible=yes);t,p=cb.begin(t,s,actor,move,self.runtime['direct_hit_context'],code=code,eligible=yes)
   after=s if code!=cb.WISH_START else s._replace(wish_counter=s.wish_counter.at[rows,actor].set(2),wish_amount=s.wish_amount.at[rows,actor].set(50));t,c=cb.finish(t,p,after)
   self.assertTrue(bool(witness._raw_complete(t,tables).all()))
   prior=t._replace(refusal=t.refusal|64)
   got=risk.self_heal_on_tape(prior,yes,active=yes,captured_self_move=c);self.assertTrue(bool((got.refusal&64).all()));self.assertEqual(((got.execution.risks.carry.risk&risk.SELF_EFFECT)!=0).tolist(),[code!=cb.REST_FAIL]*b)
   got=risk.branch_on_tape(t,risk.Branch.WISH_START,yes,active=yes,captured_self_move=c);self.assertEqual(((got.execution.risks.carry.risk&risk.FIELD_VOLATILE)!=0).tolist(),[code!=cb.WISH_START]*b)
  move=z+names.index('wish');old=s._replace(wish_counter=s.wish_counter.at[rows,actor].set(2),wish_amount=s.wish_amount.at[rows,actor].set(50));after=old._replace(wish_counter=old.wish_counter.at[rows,actor].set(1))
  t,p=cb.begin(start,old,actor,move,self.runtime['direct_hit_context'],code=cb.WISH_TICK,eligible=yes);t,c=cb.finish(t,p,after);self.assertFalse(bool(cb.captured_matches(t,c).any()));self.assertTrue(bool((t.refusal!=0).all()))

if __name__=='__main__':unittest.main()
