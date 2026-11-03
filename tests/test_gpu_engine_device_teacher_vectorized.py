"""D3 inert root/world geometry and real archive-batched model execution."""
import unittest
from unittest.mock import patch
from gpu_engine import device_public_teacher_batch as batch
from tests import test_gpu_engine_device_public_teacher_batch as fixtures
forward=fixtures.forward

class VectorizedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.FrameBatchTests.setUpClass()
    def compare(self,*args,**kwargs):
        constructor=batch.FramePricer
        def construct(*a,**kw):
            kw.update(vectorized=True,capacity=max(8,kw.get('capacity',8)))
            return constructor(*a,**kw)
        with patch.object(batch,'FramePricer',side_effect=construct):
            return fixtures.FrameBatchTests().compare(*args,**kwargs)
    def test_both_owners_mixed_archives_and_non_power_of_two_root_group(self):
        self.compare((0,0,0,0,0),(0,1,0,1,0))
    def test_status_world_and_branch_refusals_remain_literal(self):
        for mode in (1,2,3,4,5,6,7):self.compare((mode,0))
        self.compare(missing=((0,0),));self.compare(missing=((0,0),(0,1),(1,0),(1,1)))
    def test_no_sequential_map_and_real_batched_actor_rows(self):
        import jax
        shapes=[]
        def actor(*args,**kwargs):
            shapes.append(args[1]['numeric_features'].shape[0])
            return forward(*args,**kwargs)
        # This excludes every sequential root/world/leaf fallback, including
        # legacy shape-preserving maps, from the traced D3 execution path.
        with patch.object(jax.lax,'map',side_effect=AssertionError('sequential map forbidden')):
            self.compare()
        # Explicit direct trace checks peer batch>1 and leaf batch>=162.
        from tests.test_gpu_engine_device_public_teacher_batch import executor,mapping,j
        fixture=fixtures.FrameBatchTests();requests=fixture.requests();_,ex=executor()
        params=({'offset':j.asarray(0.,j.float32),'logits':j.zeros(9)},
            {'offset':j.asarray(.07,j.float32),'logits':j.zeros(9)})
        engine=batch.FramePricer(ex,object(),capacity=4,block_rows=2,actor_capacity=3,vectorized=True)
        with patch.object(mapping,'materialize',side_effect=fixture.materialize((0,0))):
            with patch.object(batch.v5_model,'forward',side_effect=actor):
                results=engine.price(requests,fixture.tables,fixture.recipe,params)
        self.assertTrue(all(r.status=='LABELED'for r in results))
        self.assertTrue(shapes);self.assertTrue(all(n>1 for n in shapes));self.assertIn(162,shapes)
        self.assertEqual(engine.last_timing['device_invocations'],1)
        self.assertEqual(engine.last_timing['padded_leaf_count'],486)
        self.assertEqual(engine.last_timing['root_capacity'],2)

if __name__=='__main__':unittest.main()
