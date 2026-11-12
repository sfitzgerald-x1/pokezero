"""Literal successful Rest same-write input components; no full kernel/game."""
import ast,json,subprocess,unittest
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_self_move_callbacks as cb,engine_execution_completeness as ex
from gpu_engine import engine_kernel_frame_risk as risk,engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_full_v4_event_device as full,engine_source_frame_witness as witness
from gpu_engine import transition as tr,public_wire,state,v4_encoder as v4
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_self_move_owning as old
BASE='2dbb27e8f61123ca75a0ef570bb4f9f8eea7fe72'

class RestSuccessTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  old.SelfMoveOwningTests.setUpClass()
  for name in ('metadata','catalog','dex','table','producer','helper','prepared','runtime'):setattr(cls,name,getattr(old.SelfMoveOwningTests,name))
  cls.fold=staticmethod(old.SelfMoveOwningTests.fold)
 def component(self,consume_initial=True):
  b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);actor=rows%2;active=j.stack((z+3,z+4),axis=1);phys=active[rows,actor]
  s=state.empty_state(b)._replace(active=active,gender=j.ones((b,2,6),j.int32),hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),level=j.full((b,2,6),80,j.int32))
  ability=self.metadata['vocabulary']['abilities'].index('pressure');item=self.metadata['vocabulary']['items'].index('leftovers')
  s=s._replace(hp=s.hp.at[rows,actor,phys].set(186),max_hp=s.max_hp.at[rows,actor,phys].set(288),ability=s.ability.at[rows,actor,phys].set(ability),item=s.item.at[rows,actor,phys].set(item))
  kw=dict(catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32));cursor=life.register(rows+97001,**kw);initial=life._tape(cursor,self.producer,64)
  for side in(0,1):initial=life._reveal(initial,s,z+side,z-1,active[:,side],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes);frame=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(cursor),life.snapshot(initial.cursor),~yes,z);consumers=[]
  for owner in((0,1)if consume_initial else()):
   stream=full.register(cursor.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=cursor.stream_id,owner_sides=z+owner);r=self.fold(stream,frame,pub);self.assertEqual(r.projection.error.tolist(),[0]*b);consumers.append((r.projection.state,r.public))
  tape=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device;tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,rt),tape,rt,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,rt)))
  return s,tape,consumers,actor,phys

 def transaction(self,s,t,actor,phys):
  b=len(actor);rows=j.arange(b);yes=j.ones(b,bool);move=j.full(b,self.metadata['vocabulary']['moves'].index('rest'),j.int32)
  t=public.move(t,actor,phys,s.active[rows,1-actor],move,eligible=yes)
  t,p=cb.begin(t,s,actor,move,self.runtime['direct_hit_context'],code=cb.REST_FAIL,eligible=yes,secondary=yes)
  return t,p

 def after(self,s,actor,phys):
  at=(j.arange(len(actor)),actor,phys)
  return s._replace(hp=s.hp.at[at].set(s.max_hp[at]),status=s.status.at[at].set(5),status_counter=s.status_counter.at[at].set(3),sleep_skipped=s.sleep_skipped.at[at].set(0),rest_origin=s.rest_origin.at[at].set(True))

 def test_same_write_guards_and_mandatory_final_accounting(self):
  s,start,_,actor,phys=self.component(False);yes=j.ones(8,bool);rows=j.arange(8);t,p=self.transaction(s,start,actor,phys)
  self.assertTrue(bool(cb.rest_write_supported(t,p,yes).all()))
  after=self.after(s,actor,phys);got,c=cb.finish(t,p,after);good_capture=c;self.assertTrue(bool(cb.captured_matches(got,c).all()))
  self.assertEqual(got.count.tolist(),[3]*8);self.assertTrue(bool(witness._raw_complete(got,self.runtime['frame_witness_compiled'].device).all()))
  at=(rows,actor,phys)
  for field,value in (('hp',287),('max_hp',289),('status',0),('status_counter',2),('sleep_skipped',1),('rest_origin',False),('ability',0),('item',0),('volatile',1)):
   bad=after._replace(**{field:getattr(after,field).at[at].set(value)})
   refused,c=cb.finish(t,p,bad);self.assertFalse(bool(cb.captured_matches(refused,c).any()),field);self.assertTrue(bool((refused.refusal!=0).all()),field)
   ledger=refused.execution.ledger;self.assertTrue(bool((ledger.stage[:,0]==1).all()),field)
  for field in ('engine_binding','source_binding','move_binding','producer_binding'):
   with self.assertRaises(ValueError):cb.rest_write_supported(t._replace(**{field:j.broadcast_to(getattr(t,field),(8,8))}),p,yes)
  for tape in (t._replace(cursor=t.cursor._replace(turn=t.cursor.turn+1)),t._replace(event_phase=t.event_phase.at[:,0].add(1)),t._replace(words=t.words.at[:,0,5].add(1))):
   self.assertFalse(bool(cb.rest_write_supported(tape,p,yes).any()))
  for bad in (s._replace(status=s.status.at[at].set(5)),s._replace(sleep_skipped=s.sleep_skipped.at[at].set(1)),s._replace(rest_origin=s.rest_origin.at[at].set(True)),s._replace(volatile=s.volatile.at[at].set(1))):
   bt,bp=self.transaction(bad,start,actor,phys);self.assertFalse(bool(cb.rest_write_supported(bt,bp,yes).any()))
  for field,family,name in (('ability','abilities','insomnia'),('item','items','lumberry')):
   bad=s._replace(**{field:getattr(s,field).at[at].set(self.metadata['vocabulary'][family].index(name))})
   bt,bp=self.transaction(bad,start,actor,phys);self.assertFalse(bool(cb.rest_write_supported(bt,bp,yes).any()),name)
   refused,bc=cb.finish(bt,bp,self.after(bad,actor,phys));self.assertFalse(bool(cb.captured_matches(refused,bc).any()),name);self.assertTrue(bool((refused.execution.ledger.refusal!=0).all()),name)
  prior=got._replace(refusal=got.refusal|64);r=risk.self_heal_on_tape(prior,yes,active=yes,captured_self_move=good_capture);self.assertTrue(bool((r.refusal&64).all()));self.assertFalse(bool((r.execution.risks.carry.risk&risk.SELF_EFFECT).any()))
  old=prior.execution.risks;old=old._replace(carry=old.carry._replace(risk=old.carry.risk|risk.SELF_EFFECT));prior=prior._replace(execution=prior.execution._replace(risks=old))
  r=risk.self_heal_on_tape(prior,yes,active=yes,captured_self_move=good_capture);self.assertTrue(bool((r.execution.risks.carry.risk&risk.SELF_EFFECT).all()))
  mixed=after._replace(status_counter=after.status_counter.at[0,actor[0],phys[0]].set(2));mt,mc=cb.finish(t,p,mixed)
  self.assertEqual(cb.captured_matches(mt,mc).tolist(),[False]+[True]*7);self.assertEqual((mt.execution.ledger.refusal!=0).tolist(),[True]+[False]*7)
  self.assertFalse(bool(cb.rest_write_supported(t,None,yes).any()))
  self.assertFalse(bool(cb.captured_matches(got._replace(count=got.count-1),good_capture).any()))
  # Available source masks are exact original immunity/Lum tables, not names.
  d=self.runtime['direct_hit_context'].data.self_move
  self.assertFalse(bool(np.any(np.asarray(d.rest_items)&np.asarray(self.table.items.lum))))
  self.assertFalse(bool(np.any(np.asarray(d.rest_abilities)&np.asarray(self.table.secondary.ability_immunity[:,5]))))
  self.assertFalse(bool(cb.captured_matches(got._replace(words=got.words.at[:,2,8].set(0)),c).any()))

 def test_source_default_and_original_handlers(self):
  witness._source();self.assertEqual(len(witness.SOURCE_PINS),88)
  oldtree=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));new=ast.parse(subprocess.check_output(['git','show','da41da40:gpu_engine/transition.py'],text=True));baseline=dump_default_program(oldtree)
  self.assertEqual(dump_default_program(new),baseline);assert_hp_rng_mutations_detected(self,new,baseline)
  new=ast.parse(Path('gpu_engine/transition.py').read_text());integrated=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/transition.py'],text=True))
  self.assertEqual(ast.dump(new),ast.dump(integrated));assert_hp_rng_mutations_detected(self,new,dump_default_program(integrated))
  script=r"""const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),{Battle}=require(process.argv[1]+'/dist/sim/battle.js'),lines=[],p={hp:186,maxhp:288,status:'',isActive:true,hasAbility:()=>false,toString:()=> 'p1a: Actor',statusState:{},removeVolatile:()=>false,getHealth:()=>`${p.hp}/${p.maxhp} slp`},b={effectState:p.statusState,random:()=>4,trunc:Math.trunc,runEvent:(...x)=>x.at(-1),add:(...x)=>lines.push(x.map(v=>typeof v==='function'?v():String(v)).join('|'))};p.heal=x=>{const old=p.hp;p.hp=Math.min(p.maxhp,p.hp+x);return p.hp-old};p.setStatus=(x,s,m)=>{p.status=x;d.conditions.get(x).onStart.call(b,p,s,m);return true};b.heal=x=>Battle.prototype.heal.call(b,x,p,p,move);const move=d.moves.get('rest');move.onHit.call(b,p,p,move);console.log(JSON.stringify({lines,time:p.statusState.time,start:p.statusState.startTime}));"""
  got=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(got['lines'],['-status|p1a: Actor|slp|[from] move: Rest','-heal|p1a: Actor|288/288 slp|[silent]']);self.assertEqual((got['time'],got['start']),(3,3))
  for modules in [('benchmark_retained_games','engine_lifecycle_event_device','engine_full_v4_event_device'),('engine_full_v4_event_device','engine_lifecycle_event_device','benchmark_retained_games')]:subprocess.run([__import__('sys').executable,'-c',';'.join('import gpu_engine.'+n for n in modules)],check=True,capture_output=True)

 def test_both_full_original_rest_status_then_silent_heal(self):
  s,start,consumers,actor,phys=self.component();t,p=self.transaction(s,start,actor,phys);yes=j.ones(8,bool);z=j.zeros(8,j.int32)
  self.assertTrue(bool(cb.rest_write_supported(t,p,yes).all()));after=self.after(s,actor,phys);t,c=cb.finish(t,p,after);self.assertTrue(bool(cb.captured_matches(t,c).all()))
  frame=life.LifecycleResult(after,t.cursor,t,life.snapshot(start.cursor),life.snapshot(t.cursor),~yes,z);capture=[]
  detail=jax.jit(lambda st,fr,pub:(full.project_frame(st,fr,self.runtime['public'].device),self.fold(st,fr,pub)))
  for owner,(stream,pub) in enumerate(consumers):
   projected,r=detail(stream,frame,pub);capture.append(dict(owner=owner,project_error=projected.error.tolist(),project_words=projected.frames.words.tolist(),project_valid=projected.frames.valid.tolist(),error=r.projection.error.tolist(),fold_error=r.public.refusal.tolist() if hasattr(r.public,'refusal') else None,words=r.projection.frames.words.tolist(),valid=r.projection.frames.valid.tolist()));Path('/tmp/pokezero-rest-success-full-literal.json').write_text(json.dumps(capture)+'\n');self.assertEqual(r.projection.error.tolist(),[0]*8)
   for row in range(8):
    ids={side:f'p{side+1}a: '+self.catalog.species.names[side+1]for side in(0,1)};who=int(actor[row]);initial=[]
    for side in(0,1):
     hp='186/288'if side==who and side==owner else '65/100'if side==who else'100/100'
     initial.append(f'|switch|{ids[side]}|{self.catalog.species.names[side+1]}, L80, M|{hp}')
    wire,_=public_wire.project_lines(public_wire.WireState(),initial+['|turn|1'],**v4.wire_kwargs(self.catalog))
    hp='288/288'if owner==who else'100/100';lines=[f'|move|{ids[who]}|Rest|{ids[who]}',f'|-status|{ids[who]}|slp|[from] move: Rest',f'|-heal|{ids[who]}|{hp} slp|[silent]']
    _,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));want=np.asarray([e.words()for p in packets for e in p.events],np.int32);actual=np.asarray(r.projection.frames.words[:,row])[np.asarray(r.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,want)
   self.assertFalse(bool(r.public.cursor.needs_more.any()))

if __name__=='__main__':unittest.main()
