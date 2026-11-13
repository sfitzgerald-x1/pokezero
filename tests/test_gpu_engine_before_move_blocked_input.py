"""Source-only PAR/FRZ BeforeMove and original BOTH FULL; no engine turn."""
import ast,json,subprocess,unittest
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_before_move_blocked_callbacks as cb,engine_execution_completeness as ex,engine_kernel_frame_risk as risk,engine_lifecycle_callbacks as life,engine_public_events as public,engine_full_v4_event_device as full,engine_source_frame_witness as witness
from gpu_engine import mechanics as m,sleep_talk as talking,v4_encoder as v4,public_wire,state,engine_direct_hit_controls as direct
from _showdown_root import showdown_root
import test_gpu_engine_self_boost_input as fixtures
import test_gpu_engine_transition_observer_wiring as static
BASE='4bc12b59c24d4d7f2f51f1f4248c966e71b74f48'
class BeforeMoveBlockedInputTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.SelfBoostInputTests.setUpClass();f=fixtures.SelfBoostInputTests;cls.f=f;cls.runtime=f.runtime;cls.helper=f.helper;cls.catalog=f.catalog;cls.producer=f.producer;cls.metadata=f.metadata
  cls.fold=staticmethod(jax.jit(lambda st,fr,p:full.project_and_fold(st,fr,cls.runtime['public'].device,p,cls.catalog)))
  cls.consume=staticmethod(jax.jit(lambda l,f,s,p:ex.consume_frame_device(l,f,s,p,active=j.arange(8)<4,tables=cls.runtime['execution_compiled'].device,adapter_tables=cls.runtime['public'].device,catalog=cls.catalog)))
 def component(self):
  b=8;r=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool);side=j.array([0,1,1,0,0,1,0,1],j.int32);active=j.stack((z+3,z+4),axis=1);physical=active[r,side];ability=self.metadata['vocabulary']['abilities'].index('pressure');statuses=j.array([m.PAR,m.FRZ,m.PAR,m.FRZ,m.PAR,m.FRZ,m.PAR,m.FRZ],j.int32)
  s=state.empty_state(b)._replace(active=active,hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),level=j.full((b,2,6),80,j.int32),gender=j.ones((b,2,6),j.int32),ability=j.full((b,2,6),ability,j.int32),base_ability=j.full((b,2,6),ability,j.int32),status=j.zeros((b,2,6),j.int32).at[r,side,physical].set(statuses),confusion_counter=j.zeros((b,2,6),j.int32).at[6,side[6],physical[6]].set(2),volatile=j.zeros((b,2,6),j.int32).at[7,side[7],physical[7]].set(1),done=~yes)
  kw=dict(catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32));start=life.register(r+234001,**kw);initial=life._tape(start,self.producer,64)
  for seat in(0,1):initial=life._reveal(initial,s,z+seat,z-1,active[:,seat],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes);t=life._tape(initial.cursor,self.producer,64);a=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device;t=ex.begin_on_tape(t,ex.initialize(start.stream_id,compiled=a),a.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(t.stream_id,rt),t,rt,active=yes);t=t._replace(execution=t.execution._replace(risks=risk.Context(carry,rt)))
  draws=j.tile(j.array([99,0,0],j.int32),(b,1)).at[4,1].set(25).at[5,0].set(19).at[6,2].set(99);loafing=j.arange(b)==7;move=z+self.metadata['vocabulary']['moves'].index('surf');before_tape=t
  t,p=cb.begin(t,s,side,move,move,draws,self.runtime['direct_hit_context'],eligible=yes,sleep_usable=~yes,loafing=loafing)
  before=m.before_move(statuses,z,s.confusion_counter[r,side,physical],thaw_roll=draws[:,0],paralysis_roll=draws[:,1],confusion_roll=draws[:,2],flinched=s.volatile[r,side,physical]!=0,loafing=loafing)
  after=talking.before_move(s,side,before.status,eligible=yes,sleep_usable=~yes)
  for name,value in [('status',before.status),('status_counter',before.counter),('confusion_counter',before.confusion_counter)]:after=after._replace(**{name:getattr(after,name).at[r,side,physical].set(value)})
  t,c=cb.finish(t,p,after,before);return SimpleNamespace(s=s,after=after,before=before,side=side,draws=draws,move=move,tape=t,capture=c,prior=before_tape,start=start,initial=initial)
 def test_same_call_mixed_guards_source_stamps_and_sticky(self):
  x=self.component();t=x.tape;c=x.capture;expected=[True]*4+[False]*4;self.assertEqual(cb.owned(t,0).tolist(),expected);self.assertEqual(cb.captured_matches(t,c,x.s,x.side,x.before).tolist(),expected)
  got=risk.pre_move_on_tape(t,x.s,x.side,x.before,eligible=j.ones(8,bool),active=j.ones(8,bool),captured_before_move=c);self.assertEqual((got.execution.risks.carry.risk&risk.BLOCKED_ACTION!=0).tolist(),[False]*6+[True,True]);self.assertEqual(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).tolist(),[True]*8)
  for n in('engine_binding','source_binding','move_binding','producer_binding'):
   self.assertFalse(bool(cb.owned(t._replace(**{n:getattr(t,n).at[0].add(j.uint32(1))}),0).any()))
   with self.assertRaises(ValueError):cb.owned(t._replace(**{n:j.broadcast_to(getattr(t,n),(8,8))}),0)
  for n in('turn','phase','generation'):
   self.assertFalse(bool(cb.captured_matches(t._replace(cursor=t.cursor._replace(**{n:getattr(t.cursor,n)+1})),c,x.s,x.side,x.before).any()))
  for bad in(t._replace(words=t.words.at[0,0,6].set(25)),t._replace(event_phase=t.event_phase.at[1,0].add(1)),t._replace(execution=t.execution._replace(ledger=t.execution.ledger._replace(claimed=t.execution.ledger.claimed.at[2,0].set(0))))):self.assertFalse(bool(cb.owned(bad,0)[:4].all()))
  mismatched=c._replace(raw=c.raw.at[0,5].add(1));self.assertFalse(bool(cb.captured_matches(t,mismatched,x.s,x.side,x.before)[0]))
  missing=risk.pre_move_on_tape(t,x.s,x.side,x.before,eligible=j.ones(8,bool),active=j.ones(8,bool));self.assertTrue(bool(j.all((missing.execution.risks.carry.risk[:4]&risk.BLOCKED_ACTION)!=0)))
  prior=t._replace(refusal=t.refusal.at[0].set(65),execution=t.execution._replace(risks=t.execution.risks._replace(carry=t.execution.risks.carry._replace(risk=t.execution.risks.carry.risk.at[0].set(72)))));kept=risk.pre_move_on_tape(prior,x.s,x.side,x.before,eligible=j.ones(8,bool),active=j.ones(8,bool),captured_before_move=c);self.assertEqual(int(kept.refusal[0])&65,65);self.assertEqual(int(kept.execution.risks.carry.risk[0])&72,72)
  for n in('pp','keys','hp','max_hp','moves','boosts','ability','item'):np.testing.assert_array_equal(getattr(x.s,n),getattr(x.after,n))
 def test_coherent_claim_must_keep_exact_pending_draws_and_ability(self):
  x=self.component();t=x.tape;c=x.capture;other=[i for i,v in enumerate(np.asarray(c.pending.context.data.before_move_blocked.abilities))if i>0 and bool(v)and i!=int(c.pending.ability[0])];self.assertTrue(other)
  for col,value in ((5,98),(6,1),(7,1),(9,other[0])):
   words=t.words.at[0,0,col].set(value);changed=t._replace(words=words,execution=t.execution._replace(ledger=t.execution.ledger._replace(words=words)));capture=c._replace(raw=c.raw.at[0,col].set(value));self.assertEqual(cb.captured_matches(changed,capture,x.s,x.side,x.before).tolist(),[False,True,True,True,False,False,False,False])
 def test_original_callbacks_default_and_legacy_closed(self):
  # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
  self.assertEqual(static._undefined_step_globals(Path('gpu_engine/transition.py').read_text()),set());self.assertIsNone(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.before_move_blocked);witness._source()
  script="""const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),out=[];for(const [status,roll,defrost,magic]of [['par',0,false,false],['par',24,false,false],['par',25,false,false],['frz',19,false,false],['frz',20,false,false],['frz',99,false,false],['frz',99,true,false],['par',0,false,true]]){const lines=[],calls=[],p={hasAbility:n=>n==='magicguard'&&magic,cureStatus:()=>lines.push('cure'),toString:()=> 'p2a: Actor'},b={randomChance:(n,z)=>{calls.push([n,z]);return roll<100*n/z;},add:(...a)=>lines.push(a.map(String).join('|'))};const r=d.conditions.get(status).onBeforeMove.call(b,p,null,{flags:{defrost}});out.push({status,roll,defrost,magic,blocked:r===false,lines,calls});}console.log(JSON.stringify(out));"""
  v=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual([r['blocked']for r in v],[True,True,False,False,True,True,False,False]);self.assertEqual(v[0]['lines'],['cant|p2a: Actor|par']);self.assertEqual(v[4]['lines'],['cant|p2a: Actor|frz']);self.assertEqual(v[0]['calls'],[[1,4]]);self.assertEqual(v[4]['calls'],[[1,5]])
 def test_both_original_full_consumers_exact_cant_packets(self):
  x=self.component();b=8;z=j.zeros(b,j.int32);yes=j.ones(b,bool);f0=life.LifecycleResult(x.s,x.initial.cursor,x.initial,life.snapshot(x.start),life.snapshot(x.initial.cursor),~yes,z);owners=[]
  for owner in (0,1):
   stream=full.register(x.start.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=x.start.stream_id,owner_sides=z+owner);out=self.fold(stream,f0,pub);self.assertEqual(out.projection.error.tolist(),[0]*b);owners.append((out.projection.state,out.public))
  frame=life.LifecycleResult(x.after,x.tape.cursor,x.tape,life.snapshot(x.initial.cursor),life.snapshot(x.tape.cursor),~yes,z);compact=[]
  for owner,(stream,pub)in enumerate(owners):
   closed,out=self.consume(x.tape.execution.ledger,frame,stream,pub);Path('/tmp/pokezero-before-move-blocked-full-'+str(owner)+'.json').write_text(json.dumps(dict(error=out.projection.error.tolist(),stage=closed.stage.tolist(),refusal=closed.refusal.tolist(),words=out.projection.frames.words.tolist(),valid=out.projection.frames.valid.tolist()))+'\n');self.assertEqual(out.projection.error[:4].tolist(),[0]*4);self.assertEqual(closed.stage[:4,0].tolist(),[3]*4)
   for row in range(4):
    who=int(x.side[row]);reason='par'if row%2==0 else'frz';names={s:self.catalog.species.names[s+1]for s in(0,1)};ids={s:f'p{s+1}a: '+names[s]for s in(0,1)};initial=[f'|switch|{ids[s]}|{names[s]}, L80, M|100/100'+(' '+reason if s==who else'')for s in(0,1)];wire,_=public_wire.project_lines(public_wire.WireState(),initial+['|turn|1'],**v4.wire_kwargs(self.catalog));line=f'|cant|{ids[who]}|{reason}';_,packet=public_wire.project_line(wire,line,**v4.wire_kwargs(self.catalog));want=np.asarray([e.words()for e in packet.events],np.int32);actual=np.asarray(out.projection.frames.words[:,row])[np.asarray(out.projection.frames.valid[:,row])];compact.append(dict(owner=owner,row=row,line=line,actual=actual.tolist(),expected=want.tolist()));Path('/tmp/pokezero-before-move-blocked-original-both-full-compact.json').write_text(json.dumps(compact)+'\n');np.testing.assert_array_equal(actual,want)
class BeforeMoveBlockedCallerTests(unittest.TestCase):
 def test_cold_import_orders(self):
  import os,sys
  for names in [('gpu_engine.benchmark_retained_games','gpu_engine.engine_before_move_blocked_callbacks','gpu_engine.engine_full_v4_event_device'),('gpu_engine.engine_full_v4_event_device','gpu_engine.engine_before_move_blocked_callbacks','gpu_engine.benchmark_retained_games')]:
   p=subprocess.run([sys.executable,'-B','-c','import importlib;'+''.join('importlib.import_module('+repr(n)+');'for n in names)],capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),timeout=30);self.assertEqual(p.returncode,0,p.stderr)
 def test_actual_pre_and_post_call_operands(self):
  step=static._step(Path('gpu_engine/transition.py').read_text());calls={n.func.attr:n for n in ast.walk(step)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='before_move_blocked_callbacks'};self.assertEqual(set(calls),{'begin','finish'});marker=object();draws=j.arange(24).reshape(2,12);yes=j.ones(2,bool);spy=SimpleNamespace(begin=lambda *a,**kw:(a,kw),finish=lambda *a:(a,{}));env=dict(before_move_blocked_callbacks=spy,_public_trace=marker,work=marker,side=marker,move=marker,caller_move=marker,chance=draws,_direct_hit_context=marker,_direct_hit_registered=True,eligible=yes,sleep_usable=~yes,loafing=~yes,_before_move_blocked_pending=marker,before=marker)
  args,kw=eval(compile(ast.Expression(calls['begin']),'actual-before-move-observer','eval'),{},env);self.assertEqual(len(args),7);np.testing.assert_array_equal(args[5],draws[:,1:4]);self.assertIs(args[1],marker);self.assertIs(args[4],marker);self.assertIs(kw['eligible'],yes)
  args,_=eval(compile(ast.Expression(calls['finish']),'actual-before-move-observer','eval'),{},env);self.assertEqual(len(args),4);self.assertIs(args[2],marker);self.assertIs(args[3],marker)
if __name__=='__main__':unittest.main()
