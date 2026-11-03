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
if __name__=='__main__':unittest.main()
