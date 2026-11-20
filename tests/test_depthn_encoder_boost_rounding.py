"""All reachable signed boost stages retain the champion's float32 bits."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import public_mon_state, v4_assembly
from pokezero import showdown as reference


class EncoderBoostRoundingTests(unittest.TestCase):
    def test_every_boost_stage_on_both_seats_and_all_stat_columns(self):
        stages = np.arange(-6, 7, dtype=np.int32)
        state = public_mon_state.initialize(len(stages))
        state = state._replace(boosts=j.broadcast_to(j.asarray(stages)[:, None, None], (13, 2, 7)))
        result = jax.jit(v4_assembly.project_own_boosts)(state)
        columns = [reference.v4_numeric_index(getattr(reference, 'NUMERIC_BOOST_' + name))
                   for name in ('ATK', 'DEF', 'SPA', 'SPD', 'SPE')]
        expected = np.broadcast_to(np.asarray([int(n) / 6 for n in stages], np.float32)[:, None, None], (13, 2, 5))
        actual = np.asarray(result.numeric)[..., columns]
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        self.assertTrue(np.asarray(result.admitted).all())


if __name__ == '__main__':
    unittest.main()
