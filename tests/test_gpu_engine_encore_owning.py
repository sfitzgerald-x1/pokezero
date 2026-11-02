"""Original Encore helper transactions; no scientific games/models."""
from _component_evidence import capture_path as component_capture_path
import ast,json,subprocess,unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_encore_callbacks as cb,engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers
from gpu_engine import transition as tr,public_belief,v4_encoder as v4,public_wire,source_provenance
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from test_gpu_engine_gameplay_stream_comparison import state as literal_state
from _showdown_root import showdown_root
import test_gpu_engine_no_damage_owning as fixture

class EncoreOwningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.NoDamageOwningTests.setUpClass();cls.f=fixture.NoDamageOwningTests;cls.metadata=deepcopy(cls.f.metadata)
  dex=load_showdown_dex_cached(showdown_root());moves=dict(cls.f.catalog.dex.moves);cls.names=('encore','raindance','sunnyday','hail');info=dex.move_info('encore');moves['encore']=info
  if 'encore' not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append('encore');cls.metadata['moves'].append(None)
  cls.move=cls.metadata['vocabulary']['moves'].index('encore');cls.metadata['moves'][cls.move]=dict(id='encore',base_power=0,type=cls.metadata['vocabulary']['types'].index('normal'),source_type='Normal',category='Status',pp=5,accuracy=100,priority=0,flags={'protect':1,'mirror':1,'bypasssub':1,'metronome':1,'failencore':1})
  cls.dex=ShowdownDex(moves,cls.f.catalog.dex.species,cls.f.catalog.dex.type_chart)
  variants={n:Gen3RandbatSpeciesUniverse(n,80,(Gen3RandbatVariant('drum-'+n,'authored',n,'authored',80,cls.names,'Pressure','Leftovers'),))for n in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-encore-source'),variants)
  labels=set(cls.f.catalog.vocabulary.tokens)|{'move:encore','belief:possible_move:encore'}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=cls.f.catalog.masks.__dict__)
  original_flags=json.loads(subprocess.check_output(['node','-e',"const d=require(process.argv[1]).Dex.mod('gen3');console.log(JSON.stringify(JSON.parse(process.argv[2]).map(n=>!!d.moves.get(n).flags.failencore)))",str(showdown_root()/'dist/sim/dex.js'),json.dumps(cls.metadata['vocabulary']['moves'])],text=True))
  for row,flag in zip(cls.metadata['moves'],original_flags):
   if row is not None:row.setdefault('flags',{})['failencore']=flag
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  cls.rules=request.load_source_rules(showdown_root(),cls.metadata);cls.req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer)
  cls.facts=own.compile_own_facts(cls.req,cls.catalog);cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.prepared=driver.prepare(cls.req,cls.facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.asarray(cls.producer.engine_binding,np.uint32))

 @classmethod
 def tearDownClass(cls):
  pass
 def test_static_default_and_bound_request_source(self):
  from gpu_engine import engine_source_frame_witness as witness
  witness._source();d=self.helper.device.encore;self.assertEqual(int(d.move_id),self.move)
  self.assertEqual(int(self.runtime['kernel_risk_compiled'].device.selected_risk[self.move]),0)
  self.assertIsNone(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.encore)
  baseline=subprocess.check_output(['git','show','6bd337da:gpu_engine/transition.py'],text=True)
  self.assertEqual(dump_default_program(ast.parse(baseline)),dump_default_program(ast.parse(Path(tr.__file__).read_text())))
  assert_hp_rng_mutations_detected(self,ast.parse(Path(tr.__file__).read_text()),dump_default_program(ast.parse(baseline)))
  from gpu_engine import encore
  self.assertIn('if _direct_hit_context is not None and _direct_hit_registered and _direct_hit_context.data.encore is not None:',Path(tr.__file__).read_text())
  np.testing.assert_array_equal(self.table.encore.move,d.installed.move)
  self.assertTrue(bool(d.installed.allowed_last[self.metadata['vocabulary']['moves'].index('surf')]))
 def component(self,*,mixed=False):
  from gpu_engine import state,encore
  b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);side=rows%2;active=j.stack((z+3,z+4),axis=1);physical=active[rows,side]
  surf=self.metadata['vocabulary']['moves'].index('surf');s=state.empty_state(b)._replace(done=j.zeros(b,bool),active=active,gender=j.ones((b,2,6),j.int32),hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),level=j.full((b,2,6),80,j.int32),ability=j.full((b,2,6),self.metadata['vocabulary']['abilities'].index('pressure'),j.int32))
  types=self.rules.device.species_types[s.species];s=s._replace(types=types,base_types=types,moves=s.moves.at[rows,side,physical,1].set(surf),pp=s.pp.at[rows,side,physical,1].set(8),last_action_kind=j.full((b,2),2,j.int32),last_action_index=j.ones((b,2),j.int32))
  eligible=yes
  if mixed:
   s=s._replace(encore_counter=s.encore_counter.at[2,side[2],physical[2]].set(2),encore_move=s.encore_move.at[2,side[2],physical[2]].set(surf),pp=s.pp.at[3,side[3],physical[3],1].set(0),moves=s.moves.at[4,side[4],physical[4],2].set(surf),last_action_kind=s.last_action_kind.at[5,side[5]].set(0));eligible=yes.at[6].set(False)
  start=life.register(rows+64001,**self.kw);initial=life._tape(start,self.producer,64)
  for seat in(0,1):initial=life._reveal(initial,s,z+seat,z-1,active[:,seat],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes)
  frame0=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(start),life.snapshot(initial.cursor),~yes,z);consumers=[]
  for owner in(0,1):
   stream=full.register(start.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=start.stream_id,owner_sides=z+owner);got=self.fold(stream,frame0,pub);self.assertEqual(got.projection.error.tolist(),[0]*8);consumers.append((got.projection.state,got.public))
  tape=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device;tape=ex.begin_on_tape(tape,ex.initialize(start.stream_id,compiled=account),account.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,rt),tape,rt,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,rt)))
  move=z+self.move;tape=public._append(tape,public.MOVE,1-side,active[rows,1-side],side,physical,(move,z,z,z,z),eligible)
  tape,p=cb.begin(tape,s,side,move,self.runtime['direct_hit_context'],self.table.encore,eligible=eligible,secondary=yes,roll=z,queued=yes)
  after=encore.apply(s,side,move,z,self.table.encore,eligible=eligible,queued=yes);tape,c=cb.finish(tape,p,after)
  return s,after,side,eligible,tape,c,consumers,initial.cursor
 @classmethod
 def setUp(cls):
  if not hasattr(cls,'fold'):
   cls.fold=staticmethod(jax.jit(lambda st,fr,p:full.project_and_fold(st,fr,cls.runtime['public'].device,p,cls.catalog)))
   cls.consume=staticmethod(jax.jit(lambda st,fr,p:ex.consume_frame_device(fr.callbacks.execution.ledger,fr,st,p,active=j.ones(len(fr.callbacks.count),bool),tables=cls.runtime['execution_compiled'].device,adapter_tables=cls.runtime['public'].device,catalog=cls.catalog)))
 def test_mixed_original_apply_claim_namespace_and_sticky(self):
  s,after,side,eligible,t,c,_,_=self.component(mixed=True);expected=[True,True,False,False,False,False,False,True]
  self.assertEqual(cb.owned(t,j.maximum(t.count-1,0)).tolist(),expected);self.assertEqual(cb.captured_matches(t,c).tolist(),expected)
  for name in('engine_binding','source_binding','move_binding','producer_binding'):
   self.assertFalse(bool(cb.owned(t._replace(**{name:getattr(t,name).at[0].add(j.uint32(1))}),j.maximum(t.count-1,0)).any()))
   with self.assertRaises(ValueError):cb.owned(t._replace(**{name:j.broadcast_to(getattr(t,name),(8,8))}),0)
  for name in('turn','phase','generation'):
   self.assertFalse(bool(cb.captured_matches(t._replace(cursor=t.cursor._replace(**{name:getattr(t.cursor,name)+1})),c).any()))
  for field in('pp','hp','moves','keys','status','ability','types'):
   np.testing.assert_array_equal(getattr(s,field),getattr(after,field))
  bad=t._replace(words=t.words.at[0,1,8].add(1));self.assertFalse(bool(cb.owned(bad,1)[0]))
  previous=t._replace(refusal=t.refusal|65);got=risk.branch_on_tape(previous,risk.Branch.ENCORE,eligible,active=j.ones(8,bool),captured_encore=c);self.assertEqual(int(got.refusal[0])&65,65)
 def test_both_full_start_tick_end_and_original_request_lock(self):
  from gpu_engine import encore,public_active,engine_request_order as ordering
  from test_gpu_engine_request_source import Arrival
  from types import SimpleNamespace
  s,after,side,eligible,t,c,consumers,cursor=self.component();b=8;z=j.zeros(b,j.int32);yes=j.ones(b,bool);rows=j.arange(b)
  # Real original residual helper, separate SAME-write ticket; no public duration.
  t,p=cb.begin(t,after,side,z,self.runtime['direct_hit_context'],self.table.encore,eligible=yes,residual=True)
  tick=encore.residual(after,side,eligible=yes);t,c2=cb.finish(t,p,tick);self.assertEqual(cb.captured_matches(t,c2).tolist(),[True]*b)
  for owner,(stream,pub)in enumerate(consumers):
   proof=cb.expected_request(tick,s,pub,owner=owner,tape=t,data=self.helper.device.encore,active_tables=self.catalog.active.device,rules=self.rules.device,healthy=yes)
   self.assertEqual(proof.slot.tolist(),j.where(side==owner,1,-1).tolist())
   stale=cb.expected_request(tick,s,pub,owner=owner,tape=t._replace(words=t.words.at[:,1,8].add(1)),data=self.helper.device.encore,active_tables=self.catalog.active.device,rules=self.rules.device,healthy=yes)
   self.assertEqual(stale.slot.tolist(),j.where(side==owner,-2,-1).tolist())
   own=z+owner;engine=j.broadcast_to(self.rules.device.engine_binding,(b,8));ledger=ordering.initialize_order(t.stream_id,own,z+6,engine_binding=engine)
   original=request.Original(*(getattr(s,key)[rows,own]for key in request.Original._fields));arrival=Arrival(t.stream_id,engine,j.zeros((b,2),j.int32),s.active,j.zeros((b,2),j.int32),j.ones((b,2),j.int32),j.ones((b,2),bool),j.take_along_axis(s.species,s.active[...,None],axis=2)[...,0]);constructor=SimpleNamespace(binding=j.ones((b,8),j.uint32))
   packet,reason=request._produce(tick,ledger,arrival,original,constructor,z.astype(bool),self.rules.device,j.ones(8,j.uint32),self.table,None,proof)
   self.assertEqual((reason&request.LOCK).tolist(),[0]*8);self.assertEqual(packet.complete.tolist(),[True]*8);self.assertEqual(packet.encore_slot.tolist(),j.where(side==owner,1,-1).tolist())
   missing,reason=request._produce(tick,ledger,arrival,original,constructor,z.astype(bool),self.rules.device,j.ones(8,j.uint32),self.table)
   self.assertEqual((reason&request.LOCK!=0).tolist(),(side==owner).tolist())
   frame=life.LifecycleResult(tick,t.cursor,t,life.snapshot(cursor),life.snapshot(t.cursor),~yes,z);drained,got=self.consume(stream,frame,pub);self.assertEqual(ex.report(drained,tables=self.runtime['execution_compiled'].device).scoped_drained.tolist(),[True]*8)
   capture_path=component_capture_path(self,'encore-component-start-capture-owner'+str(owner)+'.json');capture_path.write_text(json.dumps(dict(error=got.projection.error.tolist(),ledger_refusal=drained.refusal.tolist(),words=[np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])].tolist()for row in(0,1,7)],request_reason=reason.tolist()))+'\n')
   self.assertEqual(got.projection.error.tolist(),[0]*8);self.assertEqual(got.public.cursor.needs_more.tolist(),[False]*8)
   for row in(0,1,7):
    names={seat:self.catalog.species.names[seat+1]for seat in(0,1)};ids={seat:f'p{seat+1}a: '+names[seat]for seat in(0,1)};lines=[f'|switch|{ids[seat]}|{names[seat]}, L80, M|100/100'for seat in(0,1)]
    wire,_=public_wire.project_lines(public_wire.WireState(),lines+['|turn|1'],**v4.wire_kwargs(self.catalog));_,packet=public_wire.project_lines(wire,[f'|move|{ids[1-row%2]}|Encore|{ids[row%2]}',f'|-start|{ids[row%2]}|Encore'],**v4.wire_kwargs(self.catalog));expected=np.array([x.words()for pack in packet for x in pack.events],np.int32);actual=np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,expected)
   active=got.public.context.active;self.assertEqual(active.volatiles[rows,side,public_active.NAMES.index('encore')].tolist(),[True]*8)
   # Expiry on an original counter1 tick, after retained public start exists.
   end_before=tick._replace(encore_counter=tick.encore_counter.at[rows,side,tick.active[rows,side]].set(1));end_t=life._tape(t.cursor,self.producer,64);account=self.runtime['execution_compiled'];end_t=ex.begin_on_tape(end_t,drained,account.device,active=yes);rt=self.runtime['kernel_risk_compiled'].device;end_t=end_t._replace(execution=end_t.execution._replace(risks=risk.Context(risk.begin_frame_device(t.execution.risks.carry,end_t,rt,active=yes),rt)))
   end_t,p=cb.begin(end_t,end_before,side,z,self.runtime['direct_hit_context'],self.table.encore,eligible=yes,residual=True);ended=encore.residual(end_before,side,eligible=yes);end_t,_=cb.finish(end_t,p,ended)
   end_frame=life.LifecycleResult(ended,end_t.cursor,end_t,life.snapshot(t.cursor),life.snapshot(end_t.cursor),~yes,z);end_drained,end_got=self.consume(got.projection.state,end_frame,got.public);self.assertEqual(ex.report(end_drained,tables=account.device).scoped_drained.tolist(),[True]*8);component_capture_path(self,'encore-component-end-capture-owner'+str(owner)+'.json').write_text(json.dumps(dict(error=end_got.projection.error.tolist(),ledger_refusal=end_drained.refusal.tolist(),words=[np.asarray(end_got.projection.frames.words[:,row])[np.asarray(end_got.projection.frames.valid[:,row])].tolist()for row in(0,1,7)]))+'\n');self.assertEqual(end_got.projection.error.tolist(),[0]*8);self.assertEqual(end_got.public.context.active.volatiles[rows,side,public_active.NAMES.index('encore')].tolist(),[False]*8)
   for row in(0,1,7):
    names={seat:self.catalog.species.names[seat+1]for seat in(0,1)};ids={seat:f'p{seat+1}a: '+names[seat]for seat in(0,1)};lines=[f'|switch|{ids[seat]}|{names[seat]}, L80, M|100/100'for seat in(0,1)]
    wire,_=public_wire.project_lines(public_wire.WireState(),lines+['|turn|1',f'|move|{ids[1-row%2]}|Encore|{ids[row%2]}',f'|-start|{ids[row%2]}|Encore'],**v4.wire_kwargs(self.catalog));_,packet=public_wire.project_line(wire,f'|-end|{ids[row%2]}|Encore',**v4.wire_kwargs(self.catalog));expected=np.array([x.words()for x in packet.events],np.int32);actual=np.asarray(end_got.projection.frames.words[:,row])[np.asarray(end_got.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,expected)
class EncorePrivateAdapterTests(unittest.TestCase):
 def test_typed_request_requires_current_generation_turn_and_namespace(self):
  from types import SimpleNamespace as N
  from gpu_engine import state
  b=8;z=j.zeros(b,j.int32);rows=j.arange(b);owners=rows%2;engine=state.empty_state(b);engine=engine._replace(moves=engine.moves.at[rows,owners,0,1].set(2),pp=engine.pp.at[rows,owners,0,1].set(3),encore_move=engine.encore_move.at[rows,owners,0].set(2),encore_counter=engine.encore_counter.at[rows,owners,0].set(2),last_action_kind=j.full((b,2),2,j.int32),last_action_index=j.ones((b,2),j.int32))
  binding=j.arange(8,dtype=j.uint32);rules=N(binding=binding,engine_binding=binding);ledger=N(stream_id=rows+1,owner_side=owners,engine_binding=j.broadcast_to(binding,(b,8)),generation=z);arrival=N(generation=j.zeros((b,2),j.int32));value=cb.Request(binding,binding,rows+1,owners,z,z+2,z+1,z+2,engine.turn,arrival.generation)
  self.assertEqual(cb.request_slot(value,engine,ledger,rules,arrival)[1].tolist(),[True]*8)
  for field in ('turn','generation','stream_id','physical','counter','move','slot','owner','engine_binding','rules_binding'):
   bad=value._replace(**{field:getattr(value,field)+1});self.assertFalse(bool(cb.request_slot(bad,engine,ledger,rules,arrival)[1].any()),field)
 def test_original_private_request_and_routing_require_exact_exposure(self):
  import test_gpu_engine_engine_private_actions as fixture
  from gpu_engine import engine_private_actions as e
  fixture.EnginePrivateActionsTests.setUpClass();f=fixture.EnginePrivateActionsTests();s=f.state();ledger=f.ledger();rows=j.arange(2);owners=ledger.owner_side;active=s.active[rows,owners];forced=s.moves[rows,owners,active,0]
  s=s._replace(encore_move=s.encore_move.at[rows,owners,active].set(forced),encore_counter=s.encore_counter.at[rows,owners,active].set(2))
  source=f.source()._replace(encore_slot=j.zeros(2,j.int32),disabled=j.broadcast_to(j.array([False,True,True,True]),(2,4)))
  out=f.encode(state=s,source=source);self.assertEqual(out.admitted.tolist(),[True,True]);route=e.route_choice(out,ledger,j.zeros(2,j.int32));self.assertEqual(route.engine_action.tolist(),[0,0])
  for bad in(source._replace(encore_slot=None),source._replace(encore_slot=j.ones(2,j.int32)),source._replace(disabled=j.zeros((2,4),bool))):
   self.assertFalse(bool(f.encode(state=s,source=bad).admitted.any()))
  charged=s._replace(charge_move=s.charge_move.at[rows,owners,active].set(forced));self.assertFalse(bool(f.encode(state=charged,source=source).admitted.any()))
if __name__=='__main__':unittest.main()
