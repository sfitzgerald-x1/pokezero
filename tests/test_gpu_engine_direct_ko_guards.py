"""Bounded private typed KO guards; literal CPU fixtures only."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_direct_ko_callbacks as ko,engine_direct_hit_controls as hit
from gpu_engine import engine_execution_completeness as ex,engine_public_events as events,engine_lifecycle_callbacks as life,engine_kernel_frame_risk as risk
import test_gpu_engine_direct_ko_components as fixture

class DirectKOGuardsTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):fixture.DirectKOComponentTests.setUpClass();cls.f=fixture.DirectKOComponentTests()
 def source(self):
  f=self.f;state,cursor,_=f.initial(8,consume=False);mask=j.array([True,True]+[False]*6)
  tape=life._tape(cursor,f.producer,64);tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=f.account),f.account.device,active=j.ones(8,bool));carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,f.risks.device),tape,f.risks.device,active=j.ones(8,bool));tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,f.risks.device)))
  side=j.zeros(8,j.int32);physical=state.active[:,0];target=state.active[:,1];move=j.ones(8,j.int32)
  tape=events.move(tape,side,physical,target,move,eligible=mask)
  # Real un-clipped capacity address; the preceding valid MOVE may carry
  # switch storage. It must never suppress the overflow diagnostic.
  tape=tape._replace(words=tape.words.at[1,63].set(tape.words[1,0]),event_turn=tape.event_turn.at[1,63].set(tape.event_turn[1,0]),event_phase=tape.event_phase.at[1,63].set(tape.event_phase[1,0]),event_generation=tape.event_generation.at[1,63].set(tape.event_generation[1,0]),switch_details=tape.switch_details.at[1,63,0].set(7),count=tape.count.at[1].set(64))
  tape,ticket=ex.open_attempt(tape,ex.Site.DIRECT_HIT,1-side,active=mask)
  write=hit.HitWrite(j.full(8,40,j.int32),j.zeros(8,bool),j.ones(8,bool),j.zeros(8,bool),j.ones(8,j.int32),j.ones(8,j.int32),j.zeros(8,bool),j.zeros(8,bool),j.zeros(8,bool),j.full(8,10,j.int32),j.zeros(8,j.int32),j.full(8,100,j.int32))
  return tape,ticket,side,physical,target,move,write,state.hp[:,0,0],state.hp[:,1],mask
 def prepare(self,source,context=None):
  tape,ticket,side,physical,target,move,write,hp,party,mask=source
  return ko.prepare_controls_device(tape,ticket,side,physical,target,move,write,self.f.context if context is None else context,hp,party,eligible=mask)
 def test_b8_capacity_prior_inactive_globals_and_source_mask_cannot_be_widened(self):
  source=self.source();tape=source[0]._replace(refusal=source[0].refusal.at[0].set(65));source=(tape,*source[1:]);out,p=self.prepare(source)
  self.assertTrue(int(out.refusal[1])&events.OVERFLOW);self.assertTrue(int(out.refusal[0])&64);self.assertEqual(out.count.tolist(),tape.count.tolist())
  for before,after in zip(jax.tree.leaves(tape),jax.tree.leaves(out)):
   if getattr(before,'ndim',0) and before.shape[0]==8:np.testing.assert_array_equal(before[2:],after[2:])
  for field in ('binding','engine_binding','source_binding','move_binding','producer_binding'):
   bad=self.f.context._replace(data=self.f.context.data._replace(**{field:getattr(self.f.context.data,field)[None,:]}))
   with self.assertRaises(ValueError):self.prepare(source,bad)
  # A copied KO-mask cannot explain a different static family at outcome.
  clean=self.f.pending(2);prefix,pending=clean;outcome=risk.Outcome(pending.write.power,pending.write.critical,pending.write.hit,pending.write.immune,pending.write.numerator,pending.write.denominator,j.zeros(2,bool),pending.write.substitute,pending.write.protected,pending.write.enduring,j.zeros(2,j.int32),j.zeros(2,bool))
  changed=pending._replace(context=pending.context._replace(data=pending.context.data._replace(ko_candidates=~pending.context.data.ko_candidates)))
  self.assertFalse(bool(ko.matches_outcome_device(prefix,changed,outcome,self.f.risks.device).any()))
  changed=outcome._replace(recoil=j.ones(2,j.int32));self.assertFalse(bool(ko.matches_outcome_device(prefix,pending,changed,self.f.risks.device).any()))
  changed=outcome._replace(explodes=j.ones(2,bool));self.assertFalse(bool(ko.matches_outcome_device(prefix,pending,changed,self.f.risks.device).any()))
