"""Actual retained driver objects at the bf16 producer boundary; no model."""
import unittest
from unittest.mock import patch
import jax.numpy as j
import numpy as np
from gpu_engine import benchmark_retained_games as bench, champion
from gpu_engine import engine_retained_encoder_driver as driver, transition
import test_gpu_engine_engine_own_facts as fixtures


class RetainedThroughputCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DeviceOwnFactsTests.setUpClass()
        cls.f = fixtures.DeviceOwnFactsTests
        cls.prepared = driver.prepare(cls.f.request, cls.f.compiled, public_mode='retained-full',
            fake_forward=lambda _: driver.FakeScores(j.full(9, j.nan), j.array(99, j.int32)))

    def test_real_constructor_advance_diagnostics_and_benchmark_admission_stay_closed(self):
        config = champion.Config(width=8, heads=2, feedforward=16, layers=1)
        with patch.object(champion, 'forward', side_effect=AssertionError('no model allowed')), \
                patch.object(bench, '_network', side_effect=AssertionError('no model allowed')):
            session = driver.construct(self.f.pool, np.array([[0, 1], [0, 1]], np.int64),
                prepared=self.prepared, stream_ids=j.array([910, 911], j.int32), seed=37)
            eligible, refused, reasons = bench._admission(session.result, np.ones(2, bool), config)
            self.assertFalse(eligible.any())
            self.assertEqual(refused.tolist(), [True, True])
            self.assertTrue(np.all(reasons & driver.PUBLIC_COVERAGE))
            draws = j.zeros((2, 2, 12), j.int32).at[:, :, 4].set(1)
            chance = transition.ChanceInputs(draws, *[j.zeros((2, 2), j.int32) for _ in range(4)],
                j.zeros(2, j.int32))
            names = []
            result = driver.advance(session, j.full((2, 2), 8, j.int32), active_rows=j.array([False, True]),
                timing_callback=lambda name, ms: names.append(name), chance_inputs=chance)
            diag = driver.latest_diagnostics(session)
            self.assertEqual(diag.active_rows.tolist(), [False, True])
            self.assertEqual(diag.executed_rows.tolist(), [False, True])
            self.assertEqual(result.state.core.frame_number.tolist(), [0, 1])
            self.assertEqual(set(names), {'engine', 'requests', 'public_p1', 'public_p2',
                'encode_p1', 'encode_p2', 'admission_p1', 'admission_p2'})
            eligible, refused, reasons = bench._admission(result, np.ones(2, bool), config)
            self.assertFalse(eligible.any())
            self.assertEqual(refused.tolist(), [True, True])
            for report in result.owners:
                self.assertEqual(report.fake_calls.tolist(), [0, 0])
                self.assertFalse(bool(report.full_encoder_admitted.any()))
                self.assertFalse(bool(report.B4_qualified.any()))


if __name__ == '__main__':
    unittest.main()
