"""NON_GAME original residual setter + both original FULL writer components."""
import ast,json,subprocess,unittest
from copy import deepcopy
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_pinch_item_callbacks as cb,engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers
from gpu_engine import transition as tr,items,public_belief,v4_encoder as v4,public_wire
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from test_gpu_engine_gameplay_stream_comparison import state as literal_state
from _showdown_root import showdown_root
import test_gpu_engine_protect_owning as fixtures

class PinchItemOwningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.ProtectOwningTests.setUpClass();f=fixtures.ProtectOwningTests;cls.metadata=deepcopy(f.metadata)
  cls.berries=('liechiberry','petayaberry','salacberry')
  for n in cls.berries:
   if n not in cls.metadata['vocabulary']['items']:cls.metadata['vocabulary']['items'].append(n)
  variants={n:Gen3RandbatSpeciesUniverse(n,80,tuple(Gen3RandbatVariant('pinch-'+n+'-'+item,'authored',n,'authored',80,f.names,'Pressure',item)for item in ('Leftovers','Liechi Berry','Petaya Berry','Salac Berry')))for n in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-pinch-source'),variants)
  labels=set(f.catalog.vocabulary.tokens)|{'belief:possible_item:'+n for n in cls.berries}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),f.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=f.catalog.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=f.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  cls.rules=request.load_source_rules(showdown_root(),cls.metadata);cls.req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=f.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer)
  cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.prepared=driver.prepare(cls.req,own.compile_own_facts(cls.req,cls.catalog),public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.asarray(cls.producer.engine_binding,np.uint32))
  cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))
 def component(self):
  b=8;rows=j.arange(b);zero=j.zeros(b,j.int32);yes=j.ones(b,bool);actor=rows%2
  s=literal_state();s=jax.tree.map(lambda v:j.broadcast_to(v,(b,)+v.shape[1:])if hasattr(v,'shape')and v.ndim and v.shape[0]==1 else v,s)
  active=j.stack((zero+3,zero+4),axis=1);s=s._replace(active=active,species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),ability=j.zeros_like(s.ability),base_ability=j.zeros_like(s.base_ability),gender=j.ones_like(s.gender))
  cursor=life.register(rows+53001,**self.kw);initial=life._tape(cursor,self.producer,64)
  for side in (0,1):initial=life._reveal(initial,s,zero+side,zero-1,active[:,side],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes);frame=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(cursor),life.snapshot(initial.cursor),~yes,zero);consumers=[]
  for owner in (0,1):
   stream=full.register(cursor.stream_id,zero+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=cursor.stream_id,owner_sides=zero+owner);r=self.fold(stream,frame,pub);self.assertEqual(r.projection.error.tolist(),[0]*b);consumers.append((r.projection.state,r.public))
  tape=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];r=self.runtime['kernel_risk_compiled'];tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes)
  carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,r.device),tape,r.device,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,r.device)),refusal=j.array([1]*7+[257],j.int32))
  item=j.array([self.metadata['vocabulary']['items'].index(n)for n in ('liechiberry','petayaberry','salacberry','petayaberry','petayaberry','petayaberry','petayaberry','petayaberry')],j.int32)
  hp=j.array([25,25,25,25,26,0,25,25],j.int32);maximum=zero+100;old=j.zeros((b,7),j.int32).at[3,2].set(6);eligible=yes.at[6].set(False)
  tape,p=cb.begin(tape,s,actor,item,old,hp,maximum,self.table.items,self.runtime['direct_hit_context'],eligible=eligible)
  after_item,after_boosts=jax.jit(lambda a,z,h:items.consume_pinch(self.table.items,a,z,h,maximum,eligible=eligible))(item,old,hp)
  opened=tape
  tape=cb.finish(tape,p,after_item,after_boosts)
  return s,initial.cursor,consumers,tape,p,after_item,after_boosts,opened
 def test_original_write_mixed_boundaries_both_full_exact_words(self):
  s,cursor,consumers,tape,p,new,boosts,opened=self.component();self.assertEqual(cb.owned(tape,0).tolist(),[True,True,True,False,False,False,False,True]);self.assertEqual(tape.count.tolist(),[1,1,1,0,0,0,0,1]);self.assertEqual(new[3].item(),0);self.assertEqual(boosts[3,2].item(),6);self.assertNotEqual(int(tape.refusal[3])&256,0);self.assertEqual(int(tape.refusal[7]),257)
  clean=tape._replace(count=j.where(j.arange(8)<3,tape.count,0))
  frame=life.LifecycleResult(s,clean.cursor,clean,life.snapshot(cursor),life.snapshot(clean.cursor),j.zeros(8,bool),j.zeros(8,j.int32))
  for owner,(stream,pub)in enumerate(consumers):
   r=self.fold(stream,frame,pub);self.assertEqual(r.projection.error[:3].tolist(),[0,0,0])
   for row in range(3):
    ids={side:f'p{side+1}a: '+self.catalog.species.names[side+1]for side in (0,1)}
    wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ids[side]}|{self.catalog.species.names[side+1]}, L80, M|100/100'for side in (0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog))
    name=('Liechi Berry','Petaya Berry','Salac Berry')[row];stat=('atk','spa','spe')[row];who=row%2
    _,packets=public_wire.project_lines(wire,[f'|-enditem|{ids[who]}|{name}|[eat]',f'|-boost|{ids[who]}|{stat}|1|[from] item: {name}'],**v4.wire_kwargs(self.catalog));want=np.asarray([e.words()for p in packets for e in p.events],np.int32);actual=np.asarray(r.projection.frames.words[:,row])[np.asarray(r.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,want)
 def test_forged_current_stamps_payload_namespaces_and_write_refuse(self):
  s,cursor,consumers,tape,p,new,boosts,opened=self.component()
  for col in (1,2,3,4,5,6,7,8,9):
   bad=tape._replace(words=tape.words.at[0,0,col].add(1));self.assertFalse(bool(cb.owned(bad,0)[0]),col)
  for name in ('event_turn','event_phase','event_generation'):
   bad=tape._replace(**{name:getattr(tape,name).at[0,0].add(1)});self.assertFalse(bool(cb.owned(bad,0)[0]),name)
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   bad=tape._replace(**{name:getattr(tape,name).at[0].add(j.uint32(1))});self.assertFalse(bool(cb.owned(bad,0).any()),name)
  with self.assertRaises(ValueError):cb.owned(tape._replace(move_binding=j.broadcast_to(tape.move_binding,(8,8))),0)
  for name in ('turn','phase','generation'):
   bad=opened._replace(cursor=opened.cursor._replace(**{name:getattr(opened.cursor,name)+1}))
   result=cb.finish(bad,p,new,boosts);self.assertEqual(result.count.tolist(),[0]*8,name);self.assertTrue(bool(j.all((result.refusal[:3]&256)!=0)),name)
  result=cb.finish(opened,p,new,boosts.at[0,1].set(1));self.assertEqual(int(result.count[0]),0);self.assertNotEqual(int(result.refusal[0])&256,0)
 def test_original_default_numeric_rng_ast_and_closed_legacy(self):
  self.assertIsNone(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.pinch_item)
  old=ast.parse(subprocess.check_output(['git','show','977d524d:gpu_engine/transition.py'],text=True));after=ast.parse(Path('gpu_engine/transition.py').read_text());baseline=dump_default_program(old)
  # ONLY the new observer import is absent from the physical baseline. All
  # original imports, numerical statements and RNG arguments remain checked.
  observers=[n for n in after.body if isinstance(n,ast.ImportFrom) and n.level==1 and n.module is None and len(n.names)==1 and n.names[0].name=='engine_pinch_item_callbacks' and n.names[0].asname=='pinch_callbacks']
  self.assertEqual(len(observers),1);after.body.remove(observers[0]);self.assertEqual(baseline,dump_default_program(after));assert_hp_rng_mutations_detected(self,after,baseline)
  from gpu_engine import engine_source_frame_witness as witness
  witness._source();self.assertEqual(len(witness.SOURCE_PINS),71)
  self.assertEqual(subprocess.check_output(['git','show','977d524d:gpu_engine/items.py']),Path('gpu_engine/items.py').read_bytes())

if __name__=='__main__':unittest.main()
