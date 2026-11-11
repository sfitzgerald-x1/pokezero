"""Optional typed fixed-LEVEL Sub capture retains exact final Outcome identity."""
import unittest
import jax.numpy as j
import test_gpu_engine_seismic_substitute_input as fixtures
from gpu_engine import engine_substitute_damage_callbacks as cb,engine_public_events as public
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_direct_hit_controls as direct
class SeismicSubstituteOutcomeIdentityTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.SeismicSubstituteInputTests.setUpClass();cls.a=fixtures.SeismicSubstituteInputTests()
 def literal(self):
  a=self.a;s,t,_,actor,phys=a.component(False);rows=j.arange(8);yes=j.ones(8,bool);no=~yes;z=j.zeros(8,j.int32);target=s.active[rows,1-actor];move=z+a.metadata['vocabulary']['moves'].index('seismictoss');level=s.level[rows,actor,phys]
  t=public.move(t,actor,phys,target,move,eligible=yes)
  # Actual call shape: the routed Sub claim is open, direct hit is unrequested.
  t,subticket=ex.open_attempt(t,ex.Site.SUBSTITUTE_DAMAGE,1-actor,active=yes)
  t,directticket=ex.open_attempt(t,ex.Site.DIRECT_HIT,1-actor,active=no)
  p=cb.Pending(a.runtime['direct_hit_context'],subticket,actor,phys,target,move,yes,None,move,level)
  w=cb.Write(z,yes,no,no,no,no,no,yes,z+2,z+1,z+25,z,z+25,level,z+75,z+75);done=cb.record(t,p,w)
  o=risk.Outcome(w.power,w.critical,w.hit,w.immune,w.numerator,w.denominator,no,yes,w.protected,w.enduring,z,no)
  dw=direct.HitWrite(w.power,w.critical,w.hit,w.immune,w.numerator,w.denominator,yes,w.protected,w.enduring,w.old_hp,w.new_hp,z+100)
  paired=risk.CapturedDirectHit(p.context,directticket,actor,phys,target,move,dw)
  return done,p,w,o,paired
 def test_every_final_outcome_field_matches_with_optional_or_paired_capture(self):
  done,p,w,o,paired=self.literal();capture=cb.Captured(p,w);yes=p.requested
  self.assertTrue(bool(cb.span_matches(done,p,w).all()))
  for extra in ({},{'captured_direct':paired}):
   def result(t,final,c=capture):return risk.outcome_on_tape(t,final,eligible=yes,active=yes,captured_substitute_damage=c,**extra)
   self.assertEqual(result(done,o).execution.risks.carry.risk.tolist(),[0]*8)
   for field in o._fields:
    value=getattr(o,field);changed=~value if value.dtype==j.bool_ else value+1
    bad=result(done,o._replace(**{field:changed})).execution.risks.carry.risk
    self.assertTrue(bool((bad!=0).all()),(field,extra.keys()))
   # Row-local mismatch stays row-local; prior lifetime errors never clear.
   bad=result(done,o._replace(critical=o.critical.at[3].set(False))).execution.risks.carry.risk
   self.assertEqual((bad!=0).tolist(),[False]*3+[True]+[False]*4)
   carry=done.execution.risks.carry._replace(risk=j.full((8,),risk.SELF_EFFECT,j.int32));sticky=done._replace(execution=done.execution._replace(risks=done.execution.risks._replace(carry=carry)))
   self.assertEqual(result(sticky,o).execution.risks.carry.risk.tolist(),[risk.SELF_EFFECT]*8)
   foreign=cb.Captured(p._replace(context=p.context._replace(data=p.context.data._replace(binding=p.context.data.binding.at[0].set(p.context.data.binding[0]^j.uint32(1))))),w)
   self.assertTrue(bool((result(done,o,foreign).execution.risks.carry.risk!=0).all()))
if __name__=='__main__':unittest.main()
