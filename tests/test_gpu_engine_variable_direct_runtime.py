"""Actual variable-power writes retain the ordinary direct-hit source guards."""
import unittest
import numpy as np
from gpu_engine import engine_kernel_frame_risk as risks
import test_gpu_engine_secondary_nonproc_runtime as fixture
import gpu_engine_eager_witness as eager


class VariableDirectRuntimeTests(unittest.TestCase):
    literal_specs=(('return','Normal',0,20,'Physical'),
        ('frustration','Normal',0,20,'Physical'),
        ('icebeam','Ice',95,10,'Special'),('thunderbolt','Electric',95,15,'Special'))
    setUpClass=classmethod(fixture.SecondaryNonprocRuntimeTests.setUpClass.__func__)
    construct=fixture.SecondaryNonprocRuntimeTests.construct
    advance=fixture.SecondaryNonprocRuntimeTests.advance
    same=fixture.SecondaryNonprocRuntimeTests.same

    def test_source_resolved_return_and_frustration_reach_both_original_consumers(self):
        plain=risks.compile_risks(self.runtime['request']['lifecycle'],self.runtime['execution_compiled'])
        for action in (0,1):
            with self.subTest(action=action):
                move=self.move_ids[action]
                self.assertEqual(int(self.table.power[move]),0)
                self.assertTrue(int(plain.device.selected_risk[move])&risks.NON_WEATHER)
                self.assertTrue(bool(self.runtime['kernel_risk_compiled'].device.direct_candidates[move]))
                before,result,frame=self.advance(self.construct(),action,[[99,99],[99,99]])
                self.assertEqual(result.state.core.admitted.tolist(),[True,True])
                self.assertEqual(result.state.core.risks.risk.tolist(),[0,0])
                # The shared helper already compares every physical and RNG
                # leaf to the original, unobserved numerical transition.
                # Frustration at maximal happiness can be healed back to full
                # by Leftovers. Check the actual damage packet before that heal.
                words=np.asarray(frame.callbacks.words)
                writes=(np.arange(words.shape[1])[None,:]<np.asarray(frame.callbacks.count)[:,None])&(words[:,:,0]==2)
                self.assertTrue(bool(writes.any()))
                self.assertTrue(bool(np.all(words[:,:,5][writes]<words[:,:,6][writes])))
                for owner in result.owners:
                    self.assertEqual(owner.full_line_consumers_complete.tolist(),[True,True])
                    self.assertEqual(owner.encoded.observation.admitted.tolist(),[True,True])
                    self.assertFalse(bool(owner.scored.any()))
                    self.assertFalse(bool(owner.B4_qualified.any()))


def tearDownModule():
    eager.clear()   # drop the consume executables this module compiled


if __name__=='__main__':unittest.main()
