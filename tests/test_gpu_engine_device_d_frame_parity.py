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
if __name__=='__main__':unittest.main()
