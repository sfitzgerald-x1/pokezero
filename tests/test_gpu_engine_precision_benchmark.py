"""A3 accounting and strict saved-input admission checks; no scientific data."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PrecisionBenchmarkTests(unittest.TestCase):
    def test_policy_accounting_masks_and_kl_direction(self):
        import numpy as np
        from gpu_engine.precision_benchmark import policy_comparison
        a=np.zeros((2,9));b=a.copy();a[:,0]=10;b[:,1]=10
        legal=np.zeros_like(a,dtype=bool);legal[:,2]=True
        result=policy_comparison(a,b,legal)
        self.assertEqual(result['raw']['agreement_count'],0)
        self.assertGreater(result['raw']['kl_mean'],9)
        self.assertEqual(result['legal']['agreement'],1)
        self.assertEqual(result['legal']['kl_mean'],0)
        identical=policy_comparison(a,a,np.ones_like(legal))
        self.assertEqual(identical['raw']['kl_sum'],0)
        with self.assertRaises(ValueError):policy_comparison(a,b,np.zeros_like(legal))
        with self.assertRaises(ValueError):policy_comparison(a,b,legal.astype(int))

    def test_candidates_reproducible_and_norm_scales_initialized(self):
        import numpy as np
        from gpu_engine import champion as c
        from gpu_engine.precision_benchmark import random_parameters
        cfg=c.Config(width=8,heads=2,feedforward=16,layers=1,norm='rms',moe=True)
        a,ha=random_parameters(cfg,17);b,hb=random_parameters(cfg,17)
        self.assertEqual(ha,hb)
        self.assertNotEqual(ha,random_parameters(cfg,18)[1])
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(a['encoder.layers.0.norm1.weight'],np.ones(8))
        np.testing.assert_array_equal(a['category_embedding.weight'][0],np.zeros(8))

    def test_input_hash_rejection_precedes_loading(self):
        from gpu_engine import champion
        from gpu_engine.precision_benchmark import load_inputs
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'inputs.npz';p.write_bytes(b'not an npz')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                load_inputs(p,'0'*64,champion.Config())

    def test_gate_checks_each_target_batch_and_does_not_pool_failures(self):
        import numpy as np
        from gpu_engine.champion import Output
        from gpu_engine.precision_benchmark import precision_gate
        calls=[]
        def evaluate(dtype,args):
            rows=len(args[0]);calls.append((rows,dtype))
            logits=np.zeros((rows,9));logits[:,0]=1
            # Shape-sensitive kernel regression invisible at the smaller shape.
            if rows==8 and dtype=='bfloat16':logits[0,1]=2
            return Output(logits,np.zeros(rows),np.zeros((rows,9)))
        arrays=(np.zeros((8,23,4)),)*4;legal=np.ones((8,9),bool)
        result=precision_gate(evaluate,arrays,legal,batches=(2,4))
        self.assertEqual(calls,[(4,'float32'),(4,'bfloat16'),(8,'float32'),(8,'bfloat16')])
        self.assertTrue(result['shapes'][0]['pass'])
        self.assertFalse(result['shapes'][1]['pass'])
        self.assertFalse(result['pass'])
        with self.assertRaises(ValueError):precision_gate(evaluate,arrays,legal,batches=(5,))
