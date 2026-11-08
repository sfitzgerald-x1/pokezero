"""Prospective literal ATK/DEF setter components, no scientific games/models."""
import ast,json,subprocess,unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import gc
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_bulk_up_callbacks as cb,engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers
from gpu_engine import transition as tr,public_belief,v4_encoder as v4,public_wire,source_provenance
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from test_gpu_engine_gameplay_stream_comparison import state as literal_state
from _showdown_root import showdown_root
import test_gpu_engine_protect_owning as fixtures

class BulkUpOwningTests(unittest.TestCase):
 def setUp(self):
  self.release_compilers()
 @staticmethod
 def release_compilers(*completed):
  # Keep the exact arrays, but release completed CPU executable heaps before
  # compiling the next genuine phase. This is a correctness fixture only.
  jax.block_until_ready(completed);jax.clear_caches();gc.collect()
 def completed_phase(self,name,_milliseconds):
  if name in ('engine','requests','public_p2','admission_p2'):
   self.release_compilers()
   print(type(self).__name__+' completed CPU phase '+name,flush=True)
 def construct(self,*args,**kwargs):
  original=driver._consume
  def initial_consume(runtime,state,frame,*a,**kw):
   self.release_compilers(state,frame)
   print(type(self).__name__+' constructor inputs ready',flush=True)
   return original(runtime,state,frame,*a,**dict(kw,timing_callback=self.completed_phase))
  with patch.object(driver,'_consume',side_effect=initial_consume):
   return driver.construct(*args,**kwargs)
 @classmethod
 def setUpClass(cls):
  fixtures.ProtectOwningTests.setUpClass();cls.f=fixtures.ProtectOwningTests;cls.metadata=deepcopy(cls.f.metadata)
  dex=load_showdown_dex_cached(showdown_root());moves=dict(cls.f.dex.moves);cls.names=('bulkup','surf','protect','thunderwave');name='bulkup';info=dex.move_info(name);moves[name]=info
  if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
  cls.move=cls.metadata['vocabulary']['moves'].index(name);cls.metadata['moves'][cls.move]=dict(id=name,base_power=0,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category='Status',pp=20,accuracy=True,priority=0,flags={'snatch':1})
  cls.dex=ShowdownDex(moves,cls.f.dex.species,cls.f.dex.type_chart)
  variants={n:Gen3RandbatSpeciesUniverse(n,80,(Gen3RandbatVariant('bulk-'+n,'authored',n,'authored',80,cls.names,'Pressure','Leftovers'),))for n in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-bulk-up-source'),variants)
  labels=set(cls.f.catalog.vocabulary.tokens)|{'move:bulkup','belief:possible_move:bulkup'}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=cls.f.catalog.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  cls.rules=request.load_source_rules(showdown_root(),cls.metadata);cls.req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer)
  cls.facts=own.compile_own_facts(cls.req,cls.catalog);cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.prepared=driver.prepare(cls.req,cls.facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.asarray(cls.producer.engine_binding,np.uint32));cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))
 def test_original_source_order_caps_default_program_and_closed_compiler(self):
  from gpu_engine import engine_source_frame_witness as witness
  witness._source();self.assertEqual(len(witness.SOURCE_PINS),88);self.assertTrue(bool(self.helper.device.bulk_up.candidates[self.move]));self.assertEqual(int(self.runtime['kernel_risk_compiled'].device.selected_risk[self.move]),0)
  self.assertIsNone(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.bulk_up)
  for name in ('protect','detect','endure'):
   at=self.metadata['vocabulary']['moves'].index(name);self.assertFalse(bool(self.helper.device.bulk_up.candidates[at]))
  old=ast.parse(subprocess.check_output(['git','show','5e2a0376:gpu_engine/transition.py'],text=True));after=ast.parse(subprocess.check_output(['git','show','00307d73:gpu_engine/transition.py'],text=True));baseline=dump_default_program(old);self.assertEqual(baseline,dump_default_program(after));assert_hp_rng_mutations_detected(self,after,baseline)
  after=ast.parse(Path('gpu_engine/transition.py').read_text());integrated=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/transition.py'],text=True))
  self.assertEqual(ast.dump(after),ast.dump(integrated));assert_hp_rng_mutations_detected(self,after,dump_default_program(integrated))
  script="""const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),B=require(process.argv[1]+'/dist/sim/battle.js').Battle,P=require(process.argv[1]+'/dist/sim/pokemon.js').Pokemon,out=[];const m=d.moves.get('bulkup');
for(const stages of [[0,0],[6,0],[6,6]]){const lines=[],p={hp:100,isActive:true,boosts:{atk:stages[0],def:stages[1]},battle:{clampIntRange:(v,a,z)=>Math.max(a,Math.min(z,v))},getCappedBoost:P.prototype.getCappedBoost,boostBy:P.prototype.boostBy,toString:()=> 'p2a: Actor'};
B.prototype.boost.call({gen:3,runEvent:(id,t,s,e,b)=>b,add:(...v)=>lines.push(v.map(String).join('|'))},m.boosts,p,p,m);out.push(lines);}process.stdout.write(JSON.stringify(out));"""
  value=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(value,[['-boost|p2a: Actor|atk|1','-boost|p2a: Actor|def|1'],['-boost|p2a: Actor|atk|0','-boost|p2a: Actor|def|1'],['-boost|p2a: Actor|atk|0','-boost|p2a: Actor|def|0']])
 def component(self):
  b=8;rows=j.arange(b);yes=j.ones(b,bool);zero=j.zeros(b,j.int32);actor=rows%2
  s=jax.tree.map(j.asarray,literal_state(b));active=j.stack((zero+3,zero+4),axis=1);s=s._replace(active=active,species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),ability=j.zeros_like(s.ability),base_ability=j.zeros_like(s.base_ability),gender=j.ones_like(s.gender))
  cursor=life.register(rows+43001,**self.kw);initial=life._tape(cursor,self.producer,64)
  for side in (0,1):initial=life._reveal(initial,s,zero+side,zero-1,active[:,side],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes);frame=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(cursor),life.snapshot(initial.cursor),~yes,zero);consumers=[]
  for owner in (0,1):
   stream=full.register(cursor.stream_id,zero+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=cursor.stream_id,owner_sides=zero+owner);r=self.fold(stream,frame,pub);self.assertEqual(r.projection.error.tolist(),[0]*b);consumers.append((r.projection.state,r.public))
  tape=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];r=self.runtime['kernel_risk_compiled'];tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes)
  carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,r.device),tape,r.device,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,r.device)),refusal=j.array([1]*7+[257],j.int32))
  physical=active[rows,actor];move=zero+self.move;tape=public.move(tape,actor,physical,active[rows,1-actor],move,eligible=yes)
  old=j.zeros((b,7),j.int32).at[:,0].set(j.array([0,0,6,0,6,0,5,5])).at[:,1].set(j.array([0,0,0,6,6,0,5,-6]));eligible=yes.at[6].set(False)
  tape,p=cb.begin(tape,s,actor,move,old,self.runtime['direct_hit_context'],self.table.effects,eligible=eligible)
  after=j.clip(old+j.where((rows!=5)[:,None],self.table.effects.self_boosts[move],0),-6,6)
  tape,captured=cb.finish(tape,p,after,yes);return s,initial.cursor,consumers,tape,captured,actor
 def test_mixed_caps_nochange_inactive_sticky_and_both_full_exact_words(self):
  s,cursor,consumers,tape,captured,actor=self.component();self.assertEqual(cb.owned_pair(tape,1).tolist(),[True,True,False,False,False,False,False,True]);self.assertEqual(cb.captured_matches(tape,captured).tolist(),[True,True,False,False,False,False,False,True]);self.assertEqual(tape.count.tolist(),[3,3,1,1,1,1,1,3]);self.assertEqual(int(tape.refusal[7]),257);self.assertEqual(int(tape.refusal[6]),1);self.assertTrue(bool(j.all((tape.refusal[2:6]&256)!=0)))
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   bad=tape._replace(**{name:getattr(tape,name).at[0].add(j.uint32(1))});self.assertFalse(bool(cb.owned_pair(bad,1).any()),name)
  with self.assertRaises(ValueError):cb.owned_pair(tape._replace(move_binding=j.broadcast_to(tape.move_binding,(8,8))),1)
  for name in ('turn','phase','generation'):
   bad=tape._replace(cursor=tape.cursor._replace(**{name:getattr(tape.cursor,name)+1}));self.assertFalse(bool(cb.captured_matches(bad,captured).any()),name)
  tape=tape._replace(count=j.where(j.arange(8)<2,tape.count,0));frame=life.LifecycleResult(s,tape.cursor,tape,life.snapshot(cursor),life.snapshot(tape.cursor),j.zeros(8,bool),j.zeros(8,j.int32))
  for owner,(stream,pub)in enumerate(consumers):
   r=self.fold(stream,frame,pub);self.assertEqual(r.projection.error[:2].tolist(),[0,0])
   for row in (0,1):
    names={side:self.catalog.species.names[side+1]for side in (0,1)};ids={side:f'p{side+1}a: '+names[side]for side in (0,1)}
    wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ids[side]}|{names[side]}, L80, M|100/100'for side in (0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog));who=row%2
    _,packets=public_wire.project_lines(wire,[f'|move|{ids[who]}|Bulk Up|{ids[who]}',f'|-boost|{ids[who]}|atk|1',f'|-boost|{ids[who]}|def|1'],**v4.wire_kwargs(self.catalog));want=np.asarray([e.words()for p in packets for e in p.events],np.int32);actual=np.asarray(r.projection.frames.words[:,row])[np.asarray(r.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,want)
 def test_actual_original_frame_both_owners_witness_encoder_and_physics(self):
  arrays=self.f.f.f.base().arrays('pressure','pressure');arrays['moves'][:]=np.array([self.metadata['vocabulary']['moves'].index(n)for n in self.names]);arrays['pp'][:]=[32,24,16,32];arrays['item'][:]=0
  arrays['stats'][:,:,0]=300;arrays['stats'][:,:,3]=50;arrays['stats'][:,:,4]=200;arrays['stats'][0,:,5]=150;arrays['stats'][1,:,5]=220
  with TemporaryDirectory()as tmp:
   p=Path(tmp)/'literal.npz';np.savez_compressed(p,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.metadata).encode(),np.uint8)));pool=source_provenance.load_constructor_pool(p)
   session=self.construct(pool,np.tile([[0,1]],(2,1)).astype(np.int64),prepared=self.prepared,stream_ids=j.array([44001,44002],j.int32),seed=31)
  self.assertEqual(session.state.core.admitted.tolist(),[True,True]);before=session.state;actions=j.array([[0,1],[1,0]],j.int32);draws=j.zeros((2,2,12),j.int32).at[:,:,4].set(1);chance=tr.ChanceInputs(draws,*[j.zeros((2,2),j.int32)for _ in range(4)],j.zeros(2,j.int32));frames=[];original=self.runtime['frame']
  def capture(*a,**kw):f=original(*a,**kw);frames.append(f);return f
  self.runtime['frame']=capture
  try:result=driver.advance(session,actions,chance_inputs=chance,timing_callback=self.completed_phase)
  finally:self.runtime['frame']=original
  frame=frames[-1];plain=jax.jit(lambda state,actions:tr.step_with_chance(state,actions,self.table,chance).state)(before.core.engine,actions)
  for a,z in zip(jax.tree.leaves(frame.state),jax.tree.leaves(plain),strict=True):np.testing.assert_array_equal(a,z)
  self.assertEqual(result.state.core.admitted.tolist(),[True,True],json.dumps(dict(raw=frame.callbacks.words.tolist(),source=frame.callbacks.refusal.tolist(),risk=result.state.core.risks.risk.tolist(),ledger=result.state.core.execution.refusal.tolist(),witness=result.state.core.frame_input_report.refusal.tolist(),expected=result.state.core.frame_input_report.expected_cutpoints.tolist(),actual=result.state.core.frame_input_report.observed_cutpoints.tolist(),owners=[o.reason.tolist()for o in result.owners])))
  for row,side in ((0,0),(1,1)):
   words=np.asarray(frame.callbacks.words[row,:int(frame.callbacks.count[row])]);pair=words[words[:,0]==cb.BULK_UP];self.assertEqual(pair[:,4].tolist(),[0,1]);self.assertEqual(pair[:,5:7].tolist(),[[0,1],[0,1]]);self.assertEqual(int(frame.state.boosts[row,side,int(frame.state.active[row,side]),0]),1)
  ledger=result.state.core.execution;used=ledger.site==int(ex.Site.BULK_UP);self.assertTrue(bool(j.all(j.where(used,ledger.stage==3,True))))
  for owner in result.owners:self.assertEqual(owner.full_line_consumers_complete.tolist(),[True,True]);self.assertEqual(owner.encoded.observation.admitted.tolist(),[True,True]);self.assertFalse(bool(owner.B4_qualified.any()))

if __name__=='__main__':unittest.main()
