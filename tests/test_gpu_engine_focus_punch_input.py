"""Original Focus Punch preparation helper transactions; no engine turns."""
import ast,hashlib,json,os,subprocess,sys,unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_focus_punch_callbacks as cb,engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers
from gpu_engine import transition as tr,public_belief,v4_encoder as v4,public_wire,state,focus_punch
from gpu_engine import engine_source_frame_witness as witness
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_hidden_power_substitute_input as fixtures
import test_gpu_engine_transition_observer_wiring as static
BASE='e481228626c9cfdfe2a2989871672392aa3e5859'
class FocusPunchInputTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.HiddenPowerSubstituteInputTests.setUpClass();f=fixtures.HiddenPowerSubstituteInputTests;cls.f=f;cls.metadata=deepcopy(f.metadata)
  dex=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves);cls.names=('focuspunch','surf','icebeam','tackle')
  flags=json.loads(subprocess.check_output(['node','-e',"const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3');console.log(JSON.stringify(JSON.parse(process.argv[2]).map(n=>d.moves.get(n).flags)))",str(showdown_root()),json.dumps(cls.names)],text=True))
  for name,flag in zip(cls.names,flags):
   info=dex.move_info(name);moves[name]=info
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   i=cls.metadata['vocabulary']['moves'].index(name);cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=int(info.accuracy),priority=info.priority,flags=flag)
  cls.dex=ShowdownDex(moves,f.dex.species,f.dex.type_chart)
  variants={name:Gen3RandbatSpeciesUniverse(name,80,(Gen3RandbatVariant('focus-'+name,'unit',name,'unit',80,cls.names,'Pressure','Leftovers'),))for name in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-focus-preparation-component'),variants)
  labels=set(f.catalog.vocabulary.tokens)
  for n in cls.names:labels|={'move:'+n,'belief:possible_move:'+n}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=f.catalog.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  cls.rules=request.load_source_rules(showdown_root(),cls.metadata);req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer);facts=own.compile_own_facts(req,cls.catalog)
  cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.prepared=driver.prepare(req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.asarray(cls.producer.engine_binding,np.uint32))
  cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))
  cls.consume=staticmethod(jax.jit(lambda l,f,s,p:ex.consume_frame_device(l,f,s,p,active=j.ones(8,bool),tables=cls.runtime['execution_compiled'].device,adapter_tables=cls.runtime['public'].device,catalog=cls.catalog)))
 def component(self,*,dual=False,prior=False):
  b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);actor=rows%2;active=j.stack((z+3,z+4),axis=1);move=int(self.helper.device.focus_punch.move);surf=self.metadata['vocabulary']['moves'].index('surf');ability=self.metadata['vocabulary']['abilities'].index('pressure')
  s=state.empty_state(b)._replace(active=active,gender=j.ones((b,2,6),j.int32),hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),level=j.full((b,2,6),80,j.int32),ability=j.full((b,2,6),ability,j.int32),base_ability=j.full((b,2,6),ability,j.int32),pp=j.full((b,2,6,4),8,j.int32),done=~yes)
  selected=j.full((b,2),surf,j.int32).at[rows,actor].set(move)
  if dual:selected=selected.at[0,1-actor[0]].set(move)
  s=s._replace(moves=s.moves.at[rows[:,None],j.arange(2)[None,:],active,0].set(selected))
  cursor=life.register(rows+904101,**self.kw);initial=life._tape(cursor,self.producer,64)
  for side in (0,1):initial=life._reveal(initial,s,z+side,z-1,active[:,side],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes)
  tape=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];rc=self.runtime['kernel_risk_compiled'];tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes)
  carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,rc.device),tape,rc.device,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,rc.device)),refusal=z+1)
  tape=life.begin(tape,s,yes,~yes)
  if prior:tape=tape._replace(refusal=tape.refusal.at[0].set(257))
  before=tape;tape,p=cb.begin(tape,s,selected,j.zeros((b,2),j.int32),self.runtime['direct_hit_context'],self.table.focus_punch,ordinary=yes,move_requests=j.ones((b,2),bool))
  opened=tape
  updated=focus_punch.prepare(s,selected,self.table.focus_punch,ordinary=yes,move_requests=j.ones((b,2),bool));tape,c=cb.finish(tape,p,updated)
  return SimpleNamespace(state=s,after=updated,cursor=cursor,initial=initial,before=before,opened=opened,tape=tape,capture=c,actor=actor,move=z+move,selected=selected)
 def declare(self,x):
  return public.move(x.tape,x.actor,x.state.active[j.arange(8),x.actor],x.state.active[j.arange(8),1-x.actor],x.move,eligible=j.ones(8,bool))
 def test_source_default_original_callback_and_current_operands(self):
  witness._source();self.assertEqual(len(witness.SOURCE_PINS),86)
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));after=ast.parse(Path('gpu_engine/transition.py').read_text());base=dump_default_program(old);self.assertEqual(dump_default_program(after),base);assert_hp_rng_mutations_detected(self,after,base)
  self.assertEqual(static._undefined_step_globals(Path('gpu_engine/transition.py').read_text()),set())
  step=static._step(Path('gpu_engine/transition.py').read_text());calls={n.func.attr:n for n in ast.walk(step)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='focus_callbacks'}
  self.assertEqual(ast.unparse(calls['begin']),'focus_callbacks.begin(_public_trace, work, moves, slot, _direct_hit_context if _direct_hit_registered else None, table.focus_punch, ordinary=live & ~replacement, move_requests=(actions < 4) & ~struggle)')
  self.assertEqual(ast.unparse(calls['finish']),'focus_callbacks.finish(_public_trace, _focus_pending, work)')
  self.assertIsNone(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.focus_punch)
  script="""const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),m=d.moves.get('focuspunch'),c=d.conditions.get('focuspunch'),lines=[],p={toString:()=> 'p2a: Actor',addVolatile(id){if(id!=='focuspunch')throw Error(id);c.onStart.call({add:(...v)=>lines.push(v.map(String).join('|'))},this)}};m.beforeTurnCallback.call({},p);process.stdout.write(JSON.stringify(lines));"""
  self.assertEqual(json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True)),['-singleturn|p2a: Actor|move: Focus Punch'])
 def test_same_write_current_selection_mixed_and_refusals(self):
  x=self.component();self.assertTrue(bool(cb.owned(x.tape,1).all()));self.assertEqual(x.tape.count.tolist(),[2]*8);t=self.declare(x)
  self.assertTrue(bool(cb.matches_selected(t,x.capture,x.after,x.actor,x.move,x.move).all()))
  before=t.execution.risks.carry;out=risk.observe_selected_device(before,t,x.move,t.execution.risks.tables,eligible=j.ones(8,bool),active=j.ones(8,bool),captured_focus=x.capture,state=x.after,side=x.actor,caller_move=x.move);self.assertEqual(out.risk.tolist(),[0]*8)
  closed=risk.observe_selected_device(before,t,x.move,t.execution.risks.tables,eligible=j.ones(8,bool),active=j.ones(8,bool));self.assertTrue(bool(j.all((closed.risk&risk.BLOCKED_ACTION)!=0)))
  for name in ('hp','pp','status_counter','keys'):
   a=getattr(x.after,name);bad=x.after._replace(**{name:a.at[(0,)+(0,)*(a.ndim-1)].add(1)})
   _,c=cb.finish(x.opened,x.capture.pending,bad);self.assertFalse(bool(cb.matches_selected(t,c,x.after,x.actor,x.move,x.move)[0]))
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   self.assertFalse(bool(cb.owned(t._replace(**{name:getattr(t,name).at[0].add(j.uint32(1))}),1).any()))
   with self.assertRaises(ValueError):cb.owned(t._replace(**{name:j.broadcast_to(getattr(t,name),(8,8))}),1)
  for field in ('turn','phase','generation'):
   bad=t._replace(cursor=t.cursor._replace(**{field:getattr(t.cursor,field)+1}));self.assertFalse(bool(cb.matches_selected(bad,x.capture,x.after,x.actor,x.move,x.move).any()))
  at=(j.arange(8),x.actor,x.state.active[j.arange(8),x.actor]);lost=x.after._replace(volatile=x.after.volatile.at[at].set(j.uint32(state.FOCUS_PUNCH|state.FOCUS_LOST)));self.assertFalse(bool(cb.matches_selected(t,x.capture,lost,x.actor,x.move,x.move).any()))
  called=x.move.at[0].add(1);self.assertEqual(cb.matches_selected(t,x.capture,x.after,x.actor,x.move,called).tolist(),[False]+[True]*7)
  bad=t._replace(words=t.words.at[0,1,7].add(1));self.assertFalse(bool(cb.matches_selected(bad,x.capture,x.after,x.actor,x.move,x.move)[0]))
  for change in (dict(dual=True),dict(prior=True)):
   bad=self.component(**change);self.assertEqual(int(bad.tape.count[0]),1);self.assertEqual(int(bad.tape.execution.ledger.stage[0,1]),1);self.assertTrue(bool((bad.tape.refusal[0]&256)!=0));self.assertTrue(bool(cb.owned(bad.tape,1)[1]))
 def test_available_before_snapshot_conservation_refuses_copied_valid_bit(self):
  x=self.component();t=self.declare(x);p=x.capture.pending;who=int(p.side[0]);phys=int(p.physical[0]);at=(0,who,phys)
  changes=[dict(ability=p.before.ability.at[at].set(0),base_ability=p.before.base_ability.at[at].set(0)),dict(pending_residual=p.before.pending_residual.at[0].set(True)),dict(charge_move=p.before.charge_move.at[at].set(p.move[0]))]
  for change in changes:
   before=p.before._replace(**change);after=x.after._replace(**change);forged=p._replace(before=before);capture=x.capture._replace(pending=forged,after=after)
   self.assertEqual(cb.matches_selected(t,capture,after,x.actor,x.move,x.move).tolist(),[False]+[True]*7)
   emitted,_=cb.finish(x.opened,forged,after);self.assertEqual(cb.owned(emitted,1).tolist(),[False]+[True]*7)
   self.assertEqual(int(emitted.count[0]),1);self.assertEqual(int(emitted.execution.ledger.stage[0,1]),1);self.assertTrue(bool((emitted.refusal[0]&256)!=0))
 def test_cold_import_orders(self):
  for names in [('gpu_engine.benchmark_retained_games','gpu_engine.engine_focus_punch_callbacks','gpu_engine.engine_full_v4_event_device'),('gpu_engine.engine_full_v4_event_device','gpu_engine.engine_focus_punch_callbacks','gpu_engine.benchmark_retained_games')]:
   r=subprocess.run([sys.executable,'-B','-c','import importlib;'+''.join('importlib.import_module('+repr(n)+');'for n in names)],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30);self.assertEqual(r.returncode,0,r.stderr)
 def test_original_both_full_annotation_words_and_claims(self):
  x=self.component();b=8;z=j.zeros(b,j.int32);yes=j.ones(b,bool);initial=life.LifecycleResult(x.state,x.initial.cursor,x.initial,life.snapshot(x.cursor),life.snapshot(x.initial.cursor),~yes,z);states=[]
  for owner in (0,1):
   stream=full.register(x.cursor.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=x.cursor.stream_id,owner_sides=z+owner);r=self.fold(stream,initial,pub);self.assertEqual(r.projection.error.tolist(),[0]*b);states.append((r.projection.state,r.public))
  self.assertTrue(bool(witness._raw_complete_words(x.tape,self.runtime['frame_witness_compiled'].device)[:,:2].all()))
  frame=life.LifecycleResult(x.after,x.tape.cursor,x.tape,life.snapshot(x.initial.cursor),life.snapshot(x.tape.cursor),~yes,z);compact=[]
  for owner,(stream,pub)in enumerate(states):
   ledger,r=self.consume(x.tape.execution.ledger,frame,stream,pub)
   literal=[dict(owner=owner,row=row,actual=np.asarray(r.projection.frames.words[:,row])[np.asarray(r.projection.frames.valid[:,row])].tolist(),error=int(r.projection.error[row]),stage=ledger.stage[row,:2].tolist(),refusal=int(ledger.refusal[row]))for row in range(b)]
   Path('/tmp/pokezero-focus-punch-execution-owner-'+str(owner)+'.json').write_text(json.dumps(literal,sort_keys=True)+'\n')
   for row in range(b):
    names={side:self.catalog.species.names[side+1]for side in (0,1)};ids={side:f'p{side+1}a: '+names[side]for side in (0,1)}
    wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ids[side]}|{names[side]}, L80, M|100/100'for side in (0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog));lines=['|','|t:|1',f'|-singleturn|{ids[row%2]}|move: Focus Punch'];_,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));want=np.asarray([e.words()for p in packets for e in p.events],np.int32);actual=np.asarray(r.projection.frames.words[:,row])[np.asarray(r.projection.frames.valid[:,row])]
    compact.append(dict(owner=owner,row=row,lines=lines,actual=actual.tolist(),expected=want.tolist(),error=int(r.projection.error[row]),admitted=bool(r.projection.component_admitted[row]),needs_more=bool(r.public.cursor.needs_more[row]),stages=ledger.stage[row,:2].tolist(),refusal=int(ledger.refusal[row])))
   Path('/tmp/pokezero-focus-punch-both-full-compact.json').write_text(json.dumps(compact,sort_keys=True)+'\n')
   self.assertEqual(r.projection.error.tolist(),[0]*b);self.assertFalse(bool(r.public.cursor.needs_more.any()));self.assertTrue(bool(r.projection.component_admitted.all()));self.assertTrue(bool(j.all(ledger.stage[:,:2]==3)))
  for a in compact:np.testing.assert_array_equal(a['actual'],a['expected'])
if __name__=='__main__':unittest.main()
