"""Original ordinary sleep-block BeforeMove and BOTH FULL components only."""
import ast,json,subprocess,unittest
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_sleep_blocked_callbacks as cb,engine_execution_completeness as ex
from gpu_engine import engine_kernel_frame_risk as risk,engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_full_v4_event_device as full,mechanics as m,sleep_talk as talking,v4_encoder as v4,public_wire,state
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_self_heal_owning as fixtures
BASE='72980b7cf0315962a109d78c33f9666708892d4a'

class SleepBlockedOwningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.SelfHealOwningTests.setUpClass();cls.f=fixtures.SelfHealOwningTests;cls.runtime=cls.f.runtime;cls.helper=cls.f.helper;cls.catalog=cls.f.catalog;cls.producer=cls.f.producer;cls.metadata=cls.f.metadata
  cls.fold=staticmethod(jax.jit(lambda st,fr,p:full.project_and_fold(st,fr,cls.runtime['public'].device,p,cls.catalog)))
 def test_original_source_default_program_and_legacy_scope(self):
  from gpu_engine import engine_source_frame_witness as witness,engine_direct_hit_controls as direct
  witness._source();self.assertEqual(len(witness.SOURCE_PINS),88);self.assertIsNone(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.sleep_blocked)
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));new=ast.parse(Path('gpu_engine/transition.py').read_text());baseline=dump_default_program(old);self.assertEqual(dump_default_program(new),baseline);assert_hp_rng_mutations_detected(self,new,baseline)
  script=r'''const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),f=d.conditions.get('slp').onBeforeMove,out=[];
for(const [time,early,sleepUsable]of [[3,false,false],[2,false,false],[2,true,false],[4,true,false],[3,false,true]]){
 const lines=[],p={statusState:{time},hasAbility:id=>id==='earlybird'&&early,cureStatus(){lines.push('cure')},toString:()=> 'p2a: Actor'},b={effectState:{skippedTime:0},add:(...a)=>lines.push(a.map(String).join('|'))};
 const v=f.call(b,p,null,{sleepUsable});out.push({time:p.statusState.time,lines,blocked:v===false,skipped:b.effectState.skippedTime});}process.stdout.write(JSON.stringify(out));'''
  got=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(got[0],dict(time=2,lines=['cant|p2a: Actor|slp'],blocked=True,skipped=0));self.assertEqual(got[1]['time'],1);self.assertEqual(got[2]['lines'],['cure']);self.assertEqual(got[3]['time'],2);self.assertEqual(got[4]['skipped'],1);self.assertFalse(got[4]['blocked'])
 def component(self):
  b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);side=rows%2;active=j.stack((z+3,z+4),axis=1);physical=active[rows,side]
  s=state.empty_state(b)._replace(active=active,gender=j.ones((b,2,6),j.int32),hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),level=j.full((b,2,6),80,j.int32))
  counter=j.array([3,2,1,3,3,3,3,5],j.int32);s=s._replace(status=s.status.at[rows,side,physical].set(m.SLP),status_counter=s.status_counter.at[rows,side,physical].set(counter),rest_origin=s.rest_origin.at[3,side[3],physical[3]].set(True),confusion_counter=s.confusion_counter.at[4,side[4],physical[4]].set(2),volatile=s.volatile.at[5,side[5],physical[5]].set(1))
  kw=dict(catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32));start=life.register(rows+63001,**kw);initial=life._tape(start,self.producer,64)
  for seat in(0,1):initial=life._reveal(initial,s,z+seat,z-1,active[:,seat],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes)
  frame0=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(start),life.snapshot(initial.cursor),~yes,z);consumers=[]
  for owner in(0,1):
   stream=full.register(start.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=start.stream_id,owner_sides=z+owner);got=self.fold(stream,frame0,pub);self.assertEqual(got.projection.error.tolist(),[0]*8);consumers.append((got.projection.state,got.public))
  tape=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device;tape=ex.begin_on_tape(tape,ex.initialize(start.stream_id,compiled=account),account.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,rt),tape,rt,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,rt)))
  eligible=yes.at[6].set(False);move=z+self.metadata['vocabulary']['moves'].index('surf');table=self.producer.engine_table.sleep_talk
  tape,p=cb.begin(tape,s,side,move,self.runtime['direct_hit_context'],table,eligible=eligible,sleep_usable=~yes,early_bird=~yes,loafing=~yes)
  before=m.before_move(s.status[rows,side,physical],counter,s.confusion_counter[rows,side,physical],flinched=s.volatile[rows,side,physical]!=0)
  after=talking.before_move(s,side,before.status,eligible=eligible,sleep_usable=~yes)
  after=after._replace(status=after.status.at[rows,side,physical].set(j.where(eligible,before.status,after.status[rows,side,physical])),status_counter=after.status_counter.at[rows,side,physical].set(j.where(eligible,before.counter,after.status_counter[rows,side,physical])),confusion_counter=after.confusion_counter.at[rows,side,physical].set(j.where(eligible,before.confusion_counter,after.confusion_counter[rows,side,physical])))
  tape,c=cb.finish(tape,p,after,before)
  return s,after,before,side,eligible,tape,c,consumers,initial.cursor
 def test_actual_before_move_capture_mixed_guards_current_span_and_sticky(self):
  s,after,before,side,eligible,t,c,_,_=self.component();expected=[True,True,False,False,False,False,False,True]
  self.assertEqual(cb.owned(t,0).tolist(),expected);self.assertEqual(cb.captured_matches(t,c,s,side,before).tolist(),expected)
  from gpu_engine import engine_source_frame_witness as witness
  self.assertEqual(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).tolist(),[True]*8)
  got=risk.pre_move_on_tape(t,s,side,before,eligible=eligible,active=j.ones(8,bool),captured_sleep=c);self.assertEqual((got.execution.risks.carry.risk&risk.BLOCKED_ACTION!=0).tolist(),[False,False,False,True,True,True,False,False])
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   self.assertFalse(bool(cb.owned(t._replace(**{name:getattr(t,name).at[0].add(j.uint32(1))}),0).any()))
   with self.assertRaises(ValueError):cb.owned(t._replace(**{name:j.broadcast_to(getattr(t,name),(8,8))}),0)
  for name in('turn','phase','generation'):
   self.assertFalse(bool(cb.captured_matches(t._replace(cursor=t.cursor._replace(**{name:getattr(t.cursor,name)+1})),c,s,side,before).any()))
  for bad in(t._replace(words=t.words.at[0,0,5].add(1)),t._replace(event_turn=t.event_turn.at[0,0].add(1)),t._replace(execution=t.execution._replace(ledger=t.execution.ledger._replace(claimed=t.execution.ledger.claimed.at[0,0].set(0))))):self.assertFalse(bool(cb.owned(bad,0)[0]))
  prior=t._replace(refusal=t.refusal.at[0].set(65),execution=t.execution._replace(risks=t.execution.risks._replace(carry=t.execution.risks.carry._replace(risk=t.execution.risks.carry.risk.at[0].set(72)))))
  kept=risk.pre_move_on_tape(prior,s,side,before,eligible=eligible,active=j.ones(8,bool),captured_sleep=c);self.assertEqual(int(kept.refusal[0])&65,65);self.assertEqual(int(kept.execution.risks.carry.risk[0])&72,72)
  missing=risk.pre_move_on_tape(t,s,side,before,eligible=eligible,active=j.ones(8,bool))
  self.assertTrue(bool((missing.execution.risks.carry.risk[0]&risk.BLOCKED_ACTION)!=0))
  for field in('pp','keys','hp','max_hp','moves','boosts','ability','item'):np.testing.assert_array_equal(getattr(s,field),getattr(after,field))
 def test_both_original_full_consumers_exact_cant_sleep_packet(self):
  s,after,before,side,eligible,t,c,consumers,cursor=self.component();t=t._replace(count=j.where(j.arange(8)<2,t.count,0));frame=life.LifecycleResult(after,t.cursor,t,life.snapshot(cursor),life.snapshot(t.cursor),j.zeros(8,bool),j.zeros(8,j.int32))
  for owner,(stream,pub)in enumerate(consumers):
   got=self.fold(stream,frame,pub);self.assertEqual(got.projection.error[:2].tolist(),[0,0]);self.assertEqual(got.projection.component_admitted[:2].tolist(),[True,True])
   for row in(0,1):
    names={seat:self.catalog.species.names[seat+1]for seat in(0,1)};ids={seat:f'p{seat+1}a: '+names[seat]for seat in(0,1)};lines=[f'|switch|{ids[seat]}|{names[seat]}, L80, M|100/100'+(' slp'if seat==row else'')for seat in(0,1)]
    wire,_=public_wire.project_lines(public_wire.WireState(),lines+['|turn|1'],**v4.wire_kwargs(self.catalog));_,packet=public_wire.project_line(wire,f'|cant|{ids[row]}|slp',**v4.wire_kwargs(self.catalog));expected=np.array([x.words()for x in packet.events],np.int32);actual=np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,expected)
