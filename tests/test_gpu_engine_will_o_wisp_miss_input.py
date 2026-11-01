"""Named original Wisp miss at actual status-setter cutpoints; no engine turn."""
import ast,json,subprocess,symtable,unittest
from copy import deepcopy
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_direct_hit_controls as direct,engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own,engine_leftovers_callbacks as leftovers,transition as tr,public_belief
from _showdown_root import showdown_root
import test_gpu_engine_hidden_power_substitute_input as modern
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_primary_plain_status_callbacks as cb,engine_execution_completeness as ex
from gpu_engine import engine_kernel_frame_risk as risk,engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_source_frame_witness as witness,engine_full_v4_event_device as full
from gpu_engine import public_wire,public_inputs as inputs,v4_encoder,mechanics as m
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
import test_gpu_engine_primary_plain_status_owning as fixture
from test_gpu_engine_gameplay_stream_comparison import state as literal_state
BASE='e481228626c9cfdfe2a2989871672392aa3e5859'
class WispMissTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  modern.HiddenPowerSubstituteInputTests.setUpClass();cls.f=modern.HiddenPowerSubstituteInputTests;cls.metadata=deepcopy(cls.f.metadata);dex=load_showdown_dex_cached(showdown_root());moves=dict(cls.f.dex.moves);cls.names=('willowisp','stunspore','surf','thunderwave');labels=set(cls.f.catalog.vocabulary.tokens)
  for name in cls.names[:2]:
   info=dex.move_info(name);moves[name]=info
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   at=cls.metadata['vocabulary']['moves'].index(name);cls.metadata['moves'][at]=dict(id=name,base_power=0,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category='Status',pp=info.pp,accuracy=75,priority=0,flags={'protect':1,'reflectable':1});labels|={'move:'+name,'belief:possible_move:'+name,'move_effect:'+info.status}
  cls.dex=ShowdownDex(moves,cls.f.dex.species,cls.f.dex.type_chart);variants={n:Gen3RandbatSpeciesUniverse(n,80,(Gen3RandbatVariant('plain-'+n,'authored',n,'authored',80,cls.names,'Pressure','Leftovers'),))for n in cls.metadata['vocabulary']['species'][1:]};source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-primary-plain-source'),variants)
  cls.catalog=v4_encoder.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=cls.f.catalog.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False));cls.rules=request.load_source_rules(showdown_root(),cls.metadata);cls.req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer)
  cls.facts=own.compile_own_facts(cls.req,cls.catalog);cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.prepared=driver.prepare(cls.req,cls.facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.asarray(cls.producer.engine_binding,np.uint32));cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))
  cls.f=cls
 def component(self,b=8,*,state_change=None,attempt_change=None,caller=None,active=None,before_change=None):
  f=self.f;rows=j.arange(b);side=(rows%2).astype(j.int32);z=j.zeros(b,j.int32);yes=j.ones(b,bool);no=~yes
  state=jax.tree.map(j.asarray,literal_state(b));physical=j.full(b,3,j.int32);target=j.full(b,4,j.int32)
  act=j.zeros((b,2),j.int32).at[rows,side].set(physical).at[rows,1-side].set(target)
  state=state._replace(active=act,ability=j.zeros_like(state.ability),item=j.zeros_like(state.item),status=j.zeros_like(state.status),status_counter=j.zeros_like(state.status_counter))
  if before_change:state=before_change(state,side,physical,target)
  cursor=life.register(rows+51001,**f.kw);mapping=j.full((b,2,6),-1,j.int32).at[rows,side,physical].set(0).at[rows,1-side,target].set(0)
  cursor=cursor._replace(generation=j.ones((b,2),j.int32),turn=j.ones(b,j.int32),initialized=yes,physical_active=act,public_slot=j.zeros((b,2),j.int32),encounter=j.ones((b,2),j.int32),established=j.ones((b,2),bool),physical_to_public=mapping,next_slot=j.ones((b,2),j.int32),revealed_species_id=j.broadcast_to(j.array([1,2],j.int32),(b,2)))
  tape=life._tape(cursor,f.producer,64);account=f.runtime['execution_compiled'];r=f.runtime['kernel_risk_compiled'];tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes)
  carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,r.device),tape,r.device,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,r.device)))
  move=j.full(b,f.metadata['vocabulary']['moves'].index('willowisp'),j.int32);active=yes if active is None else active
  tape=public.move(tape,side,physical,target,move,eligible=active)
  tape,p=cb.begin(tape,state,side,move,f.runtime['direct_hit_context'],f.table,eligible=active,caller_move=move if caller is None else caller)
  attempt=cb.Attempt(no,no,no,no,no,yes,no,no,no,side==1,no,z,z,z,z)
  if attempt_change:attempt=attempt._replace(**attempt_change)
  after=state if state_change is None else state_change(state)
  tape,c=cb.finish(tape,p,attempt,state=after)
  return tape,c,state
 def test_typed_current_no_write_guards_and_same_receipt(self):
  tape,c,state=self.component();self.assertEqual(tape.count.tolist(),[2]*8);self.assertEqual(cb.captured_matches(tape,c).tolist(),[True]*8)
  self.assertEqual(cb.owned_miss(tape,1).tolist(),[True]*8);self.assertEqual(witness._raw_complete(tape,self.f.runtime['frame_witness_compiled'].device).tolist(),[True]*8)
  no=j.zeros(8,bool);yes=~no
  for name in ('accuracy_hit','hit','type_immune','ability_immune','substitute_block','applied','protected','absorbed','secondary_active'):
   bad,_,_=self.component(attempt_change={name:yes});self.assertTrue(bool(j.all((bad.refusal&256)!=0)),name)
  for field in cb._PRIVATE:
   def change(s,field=field):
    a=getattr(s,field);return s._replace(**{field:(~a if a.dtype==j.bool_ else a+1)})
   bad,_,_=self.component(state_change=change);self.assertTrue(bool(j.all((bad.refusal&256)!=0)),field)
  bad,_,_=self.component(caller=j.zeros(8,j.int32));self.assertFalse(bool(cb.owned_miss(bad,1).any()))
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   bad=tape._replace(**{name:getattr(tape,name).at[0].add(j.uint32(1))});self.assertFalse(bool(cb.owned_miss(bad,1).any()))
  for label,bad in [('claim',tape._replace(execution=tape.execution._replace(ledger=tape.execution.ledger._replace(claimed=tape.execution.ledger.claimed.at[0,1].set(0))))),('prefix',tape._replace(words=tape.words.at[0,0,5].set(0))),('clock',tape._replace(event_turn=tape.event_turn.at[0,1].add(1)))]:self.assertEqual(cb.owned_miss(bad,1).tolist(),[False]+[True]*7,label)
  bad=c._replace(after=tuple(a.at[0].set(0) for a in c.after));self.assertFalse(bool(cb.captured_matches(tape,bad)[0]))
 def test_source_default_actual_observer_operands_and_legacy(self):
  f=self.f;self.assertEqual(int(f.helper.device.primary_plain_status.wisp_miss),f.metadata['vocabulary']['moves'].index('willowisp'));witness._source()
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));new=ast.parse(Path('gpu_engine/transition.py').read_text());baseline=dump_default_program(old);self.assertEqual(baseline,dump_default_program(new));assert_hp_rng_mutations_detected(self,new,baseline)
  # Whole production function symbol table catches observer-only undefined names.
  code=Path('gpu_engine/transition.py').read_text();symbols=symtable.symtable(code,'transition.py','exec');globals_={s.get_name()for s in symbols.get_symbols()}
  import builtins
  step=next(x for x in symbols.get_children() if x.get_name()=='step_with_chance')
  self.assertFalse({s.get_name()for s in step.get_symbols() if s.is_global() and s.is_referenced()}-globals_-set(dir(builtins)))
  for filename in ('effects.py','secondary.py','state.py','mechanics.py','engine_retained_encoder_driver.py'):
   self.assertEqual(Path('gpu_engine',filename).read_bytes(),subprocess.check_output(['git','show',BASE+':gpu_engine/'+filename]))
  # No source capability means the named new route stays closed.
  tape,c,_=self.component();data=c.pending.context.data;plain=data.primary_plain_status._replace(wisp_miss=j.array(0,j.int32));context=c.pending.context._replace(data=data._replace(primary_plain_status=plain));bad=c._replace(pending=c.pending._replace(context=context));self.assertFalse(bool(cb.captured_matches(tape,bad).any()))
 def test_original_miss_writer_both_owners_join_and_atomic_rollback(self):
  # Original Gen3 tryMoveHit with inert status/accuracy event hooks. The
  # original randomChance is observed at the call boundary, never sampled.
  script=r'''const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),P=require(root+'/dist/sim/pokemon.js').Pokemon,tryHit=require(root+'/dist/data/mods/gen3/scripts.js').Scripts.actions.tryMoveHit,lines=[],events=[],a={boosts:{accuracy:0,evasion:0},toString:()=> 'p1a: Actor'},t={boosts:{accuracy:0,evasion:0},toString:()=> 'p2a: Target',runImmunity:P.prototype.runImmunity};const battle={setActiveMove(){},singleEvent:()=>true,runEvent:(id,t,s,m,value)=>{events.push(id);return value===undefined?true:value},clampIntRange:(x,lo,hi)=>Math.max(lo,Math.min(x,hi)),randomChance:(a,b)=>{events.push(['actual_accuracy',a,b]);return false},attrLastMove:x=>lines.push(['attr',x]),add:(...xs)=>lines.push(xs.map(String))};const result=tryHit.call({battle},t,a,d.moves.get('willowisp'));process.stdout.write(JSON.stringify({result,lines,events}));'''
  original=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertFalse(original['result']);self.assertEqual(original['lines'],[['attr','[miss]'],['-miss','p1a: Actor','p2a: Target']]);self.assertIn(['actual_accuracy',75,100],original['events']);self.assertNotIn('TryHit',original['events'])
  f=self.f;tape,c,state=self.component();b=8;yes=j.ones(b,bool);z=j.zeros(b,j.int32);cursor=tape.cursor
  self.assertEqual(cb.captured_matches(tape,c).tolist(),[True]*b)
  # Primary-risk closure consumes the SAME captured accuracy-stage transaction.
  tape=risk.primary_status_on_tape(tape,yes,active=yes,captured_primary_plain=c);self.assertEqual(tape.execution.risks.carry.risk.tolist(),[0]*b)
  tape=life.action_end(tape,yes);state=state._replace(done=~yes)
  tape=life.finish(tape,state,live=yes,replacement=~yes,pending_before=~yes,residual_ready=yes,defer_residual=~yes,turn_context=yes)
  self.assertEqual(cb.owned_miss(tape,1).tolist(),[True]*b)
  frame=life.LifecycleResult(state,tape.cursor,tape,life.snapshot(cursor),life.snapshot(tape.cursor),~yes,z)
  ns={k:np.asarray(getattr(f.runtime['public'].device.namespaces,k))for k in full._NAMES};captures=[];ledgers=[]
  names=[f.catalog.species.names[int(f.runtime['public'].device.causal.species[i])]for i in (1,2)];ident=[f'p{i+1}a: '+names[i]for i in (0,1)]
  prefix=[f'|switch|{ident[i]}|{names[i]}, L80, M|100/100'for i in (0,1)]+['|turn|1']
  wire,packets=public_wire.project_lines(public_wire.WireState(),prefix,**v4_encoder.wire_kwargs(f.catalog));initial=[e for p in packets for e in p.events]
  consume=jax.jit(lambda frame,stream,pub:ex.consume_frame_device(frame.callbacks.execution.ledger,frame,stream,pub,active=yes,tables=f.runtime['execution_compiled'].device,adapter_tables=f.runtime['public'].device,catalog=f.catalog))
  for owner in(0,1):
   p=v4_encoder.initialize(f.catalog,stream_ids=tape.stream_id,owner_sides=z+owner)
   for chunk in inputs.pack_chunks(np.asarray(tape.stream_id),[initial]*b,**ns):
    p,error=v4_encoder.fold_chunk(p,chunk,f.catalog);self.assertFalse(bool(error.any()))
   stream=full.register(tape.stream_id,z+owner,compiled=f.runtime['public']);stream=stream._replace(causal=stream.causal._replace(cursor=cursor,sequence=z+3),sequence=p.cursor.sequence,live=j.ones((b,2),bool),species=p.mons.species,status=p.mons.status,hp=p.mons.hp_numerator,maximum=p.mons.hp_denominator)
   ledger,got=consume(frame,stream,p);self.assertEqual(got.projection.error.tolist(),[0]*b);self.assertTrue(bool(ex.report(ledger,tables=f.runtime['execution_compiled'].device).scoped_drained.all()));ledgers.append(ledger)
   for row in range(b):
    a=row%2;lines=[f'|move|{ident[a]}|Will-O-Wisp|{ident[1-a]}|[miss]',f'|-miss|{ident[a]}|{ident[1-a]}','|','|upkeep','|turn|2']
    _,packets=public_wire.project_lines(wire,lines,**v4_encoder.wire_kwargs(f.catalog));want=np.array([e.words()for p in packets for e in p.events],np.int32);actual=np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,want);captures.append(dict(owner=owner,row=row,expected=want.tolist(),actual=actual.tolist()))
   # Corrupt one claimed raw value only. The original consumer rolls back both
   # public and transport state on precisely that row; seven other rows drain.
   badt=tape._replace(words=tape.words.at[0,1,6].set(cb.primary.LIVE|cb.primary.PROTECTED));badf=frame._replace(callbacks=badt)
   _,bad=consume(badf,stream,p);self.assertNotEqual(int(bad.projection.error[0]),0);self.assertEqual(bad.projection.error[1:].tolist(),[0]*7)
   for old,new in zip(jax.tree.leaves(stream),jax.tree.leaves(bad.projection.state),strict=True):np.testing.assert_array_equal(old[0],new[0])
   for old,new in zip(jax.tree.leaves(p),jax.tree.leaves(bad.public),strict=True):np.testing.assert_array_equal(old[0],new[0])
  joined,good=ex.join_consumed_owners(*ledgers,tables=f.runtime['execution_compiled'].device);self.assertEqual(good.tolist(),[True]*b);self.assertTrue(bool(ex.report(joined,tables=f.runtime['execution_compiled'].device).scoped_drained.all()))
  Path('/Users/scott/workspace/agents/pokezero-b4-driver-independent-review-evidence-20261030/will-o-wisp-miss-original-both-full-compact.json').write_text(json.dumps(dict(original=original,captures=captures),sort_keys=True)+'\n')

 def test_final_liveness_missing_snapshot_success_and_row_isolation(self):
  def dead(s,side,physical,target):return s._replace(hp=s.hp.at[j.arange(8),side,physical].set(0))
  bad,_,_=self.component(before_change=dead);self.assertTrue(bool(j.all((bad.refusal&256)!=0)))
  tape,c,_=self.component();self.assertFalse(bool(cb.captured_matches(tape,c._replace(pending=c.pending._replace(before=None))).any()));self.assertFalse(bool(cb.captured_matches(tape,c._replace(after=None)).any()));self.assertFalse(bool(cb.captured_matches(tape,c._replace(pending=c.pending._replace(caller=None))).any()))
  active=j.array([True]*6+[False]*2);t,c,_=self.component(active=active);self.assertEqual(t.count.tolist(),[2]*6+[0]*2);self.assertEqual(cb.captured_matches(t,c).tolist(),active.tolist())
  sticky=t._replace(refusal=t.refusal|64);observed=risk.primary_status_on_tape(sticky,active,active=active,captured_primary_plain=c);self.assertTrue(bool(j.all((observed.refusal&64)!=0)))
  # Packed int32 counter overflow must never alias contradictory actual input
  # into an emitted receipt; the other seven rows remain supported.
  bad,_,_=self.component(attempt_change={'new_counter':j.zeros(8,j.int32).at[0].set(1<<28)})
  self.assertEqual(bad.count.tolist(),[1]+[2]*7);self.assertEqual(cb.owned_miss(bad,1).tolist(),[False]+[True]*7)
  # Original successful pair remains its original two-word continuation even
  # when the new optional before/after miss inputs are omitted.
  _,_,_,old,capture=fixture.PrimaryPlainStatusTests.component(self,consume=False)
  self.assertEqual(cb.owned_pair(old,2).tolist(),[True,True,False,False,False,False,False,True]);self.assertEqual(cb.captured_matches(old,capture).tolist(),[True,True,False,False,False,False,False,True])
