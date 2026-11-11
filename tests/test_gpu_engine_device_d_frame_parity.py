"""Fixed tolerance, missing structure and refusal fields cannot disappear."""
import unittest
from gpu_engine import device_d_frame_parity as p
class ParityTests(unittest.TestCase):
    def test_float_tolerance_and_literal_unknowns(self):
        self.assertEqual(p.differences({'q':[None,.3]},{'q':[None,.3000001]}),[])
        for a,b in ((None,0.),(False,0),({'q':1.},{'q':1.,'missing':None}),([1],[1,2]),(.3,.3001),(float('nan'),float('nan'))):
            self.assertTrue(p.differences(a,b))
    def test_explicit_enablement_precedes_any_actor_or_bank_access(self):
        with self.assertRaises(ValueError):p.run(None,None,None,None,execution_source='a'*40)
    def test_numeric_evidence_does_not_hide_target_or_nonfinite_mismatches(self):
        a={'improvement':[.2,.3],'reply_logits':[1.,float('nan')]}
        b={'improvement':[.2,.30001],'reply_logits':[1.002,float('nan')]}
        r=p.numeric_summary(a,b)
        self.assertEqual(r['root_target']['above_tolerance'],1)
        self.assertAlmostEqual(r['reply_logits']['max_absolute_difference'],.002)
        self.assertEqual(r['reply_logits']['nonfinite'],1)
        self.assertEqual(len(p.differences(a,b)),3)
class D3TargetTests(unittest.TestCase):
    def pair(self):
        from dataclasses import replace
        from gpu_engine import device_public_teacher as serial
        from pokezero import v5_gumbel as g
        legal=(True,True,*([False]*7));grid=[[None]*9 for _ in range(9)]
        for a in (0,1):
            for r in (0,1):grid[a][r]=.2+.1*a
        result=g.reduce_completed_q(root_logits=(0.,)*9,root_legal=legal,worlds=(g.WorldPrices(True,legal,(0.,)*9,grid),)*2)
        return serial.Result('LABELED',result,(),(),0,None,{},())
    def test_float_level_q_and_logits_are_descriptive_not_structure_failures(self):
        from dataclasses import replace
        a=self.pair();im=a.reduction.improvement
        changed=replace(im,root_value=im.root_value+.019)
        b=replace(a,reduction=replace(a.reduction,improvement=changed))
        errors,m=p.vectorized_comparison(a,b)
        self.assertFalse(errors);self.assertAlmostEqual(m['value_target_max_absolute_difference'],.0095)
        self.assertTrue(m['top_target_action_match'])
    def test_wrong_status_and_target_outside_declared_tolerance_fail(self):
        from dataclasses import replace
        a=self.pair();b=replace(a,status='REFUSED')
        self.assertTrue(p.vectorized_comparison(a,b)[0])
        changed=replace(a.reduction.improvement,root_value=a.reduction.improvement.root_value+.021)
        b=replace(a,reduction=replace(a.reduction,improvement=changed))
        self.assertIn('value_target:absolute_tolerance',p.vectorized_comparison(a,b)[0])
    def test_one_top_action_difference_is_reported_separately(self):
        from dataclasses import replace
        a=self.pair();im=replace(a.reduction.improvement,policy_target=(.5,.5,*([0.]*7)),completed_q=(.2,.2,*([None]*7)),root_value=.2)
        a=replace(a,reduction=replace(a.reduction,improvement=im));policy=list(im.policy_target)
        i=max(range(9),key=lambda i:policy[i]);other=1-i
        # A tie-sized top change remains visible even within the float gate.
        policy[i]=.499;policy[other]=.501
        b=replace(a,reduction=replace(a.reduction,improvement=replace(im,policy_target=tuple(policy))))
        errors,m=p.vectorized_comparison(a,b)
        self.assertFalse(errors);self.assertFalse(m['top_target_action_match'])

if __name__=='__main__':unittest.main()
