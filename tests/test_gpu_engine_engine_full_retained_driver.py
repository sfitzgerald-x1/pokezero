"""Actual full-format driver integration; fake forwarding only."""
import json
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_retained_encoder_driver as driver, transition
import test_gpu_engine_engine_own_facts as fixtures


class FullRetainedDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DeviceOwnFactsTests.setUpClass()
        cls.f = fixtures.DeviceOwnFactsTests
        cls.prepared = driver.prepare(cls.f.request, cls.f.compiled,
            public_mode='retained-full', fake_forward=lambda values:
                driver.FakeScores(j.full((9,), j.nan), j.array(17, j.int32)))

    def construct(self):
        return driver.construct(self.f.pool, np.array([[0, 1], [0, 1]], np.int64),
            prepared=self.prepared, stream_ids=j.array([601, 602], j.int32), seed=37)

    def chance(self):
        draws = j.zeros((2, 2, 12), j.int32).at[:, :, 4].set(1)
        return transition.ChanceInputs(draws, *[j.zeros((2, 2), j.int32) for _ in range(4)],
            j.zeros(2, j.int32))

    def test_initial_full_controls_reach_actual_writer_without_global_promotion(self):
        session = self.construct()
        self.assertEqual(driver._checked(self.prepared)['public_mode'], 'retained-full')
        reports = []
        for observer, report in zip(session.state.observers, session.result.owners):
            self.assertEqual(observer.retained.cursor.sequence.tolist(), [18, 18])
            self.assertEqual(observer.projection.sequence.tolist(), [18, 18])
            self.assertEqual(report.retained_error.tolist(), [0, 0])
            self.assertTrue(bool(observer.retained_healthy.all()))
            self.assertTrue(bool(report.own.admitted.all()))
            # Full component formatting is not a claim of global coverage or
            # scientific qualification. The existing policy gate stays closed.
            self.assertFalse(bool(report.full_line_consumers_complete.any()))
            self.assertFalse(bool(report.full_encoder_admitted.any()))
            self.assertFalse(bool(report.B4_qualified.any()))
            self.assertFalse(bool(report.scored.any()))
            self.assertEqual(report.fake_calls.tolist(), [0, 0])
            reports.append(dict(owner=observer.owner.tolist(),
                public_error=report.retained_error.tolist(),
                unwritten_numeric=report.unwritten_numeric.tolist(),
                unwritten_categorical=report.unwritten_categorical.tolist(),
                pending_numeric_columns=report.encoded.pending_numeric_columns.tolist(),
                policy_reason=report.reason.tolist()))
        print('ACTUAL_FULL_DRIVER_WRITER_COVERAGE', json.dumps(reports, sort_keys=True))

    def test_real_core_step_is_not_duplicated_by_two_full_observers(self):
        session = self.construct()
        before = session.state.core.engine
        actions = j.zeros((2, 2), j.int32)
        chance = self.chance()
        table = driver._checked(self.prepared)['request']['engine_table']
        expected = transition.step_with_chance(before, actions, table, chance).state
        actual = driver.advance(session, actions, chance_inputs=chance)
        for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual.state.core.engine)):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(actual.state.core.frame_number.tolist(), [1, 1])
        for observer, report in zip(actual.state.observers, actual.owners):
            self.assertEqual(report.retained_error.tolist(), [0, 0])
            self.assertTrue(bool((observer.retained.cursor.sequence > 18).all()))
            self.assertFalse(bool(report.scored.any()))

    def test_full_static_holder_drift_rejects_before_any_advance(self):
        session = self.construct()
        handle = driver._checked(self.prepared)['public']
        original = handle.device
        state = session.state
        try:
            object.__setattr__(handle, 'device', original._replace(
                templates=original.templates.at[0, 0, 0].set(123)))
            with self.assertRaises(ValueError):
                driver.advance(session, j.zeros((2, 2), j.int32), chance_inputs=self.chance())
            self.assertIs(session.state, state)
        finally:
            object.__setattr__(handle, 'device', original)
        driver._checked(self.prepared)

    def test_invalid_public_modes_do_not_silently_select_a_path(self):
        for mode in (None, True, 'full', 'retained-full '):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                driver.prepare(self.f.request, self.f.compiled, public_mode=mode,
                    fake_forward=lambda _: None)

    def test_role_defaults_are_not_missing_writes_but_unwritten_nonzero_refuses(self):
        session = self.construct()
        core, observer = session.state.core, session.state.observers[0]
        facts, encoded = driver._checked(self.prepared)['encode_fast'](core, observer)
        projection = SimpleNamespace(error=j.zeros(2, j.int32),
            full_line_consumers_complete=j.ones(2, bool))
        # Synthetic component coverage exercises the pure fake logical gate;
        # actual global coverage and scientific flags are never promoted.
        finite = lambda _: driver.FakeScores(j.ones(9, j.float32), j.array(1, j.int32))
        def score(value):
            return driver._score(observer, facts, value, projection, j.zeros(2, j.int32),
                core.admitted, finite)[1]
        accepted = score(encoded)
        self.assertTrue(bool((accepted.unwritten_numeric > 0).all()))
        self.assertTrue(bool((accepted.unwritten_categorical > 0).all()))
        self.assertEqual(accepted.reason.tolist(), [0, 0])
        self.assertTrue(bool(accepted.scored.all()))
        self.assertFalse(bool(accepted.full_encoder_admitted.any()))
        self.assertFalse(bool(accepted.B4_qualified.any()))
        for values, written, name in ((encoded.observation.numeric_features,
                encoded.observation.numeric_written, 'numeric_features'),
                (encoded.observation.categorical_ids,
                 encoded.observation.categorical_written, 'categorical_ids')):
            index = tuple(np.argwhere(np.asarray(encoded.observation.attention_mask[:, :, None]
                & ~written))[0])
            bad = encoded._replace(observation=encoded.observation._replace(
                **{name: values.at[index].set(1)}))
            refused = score(bad)
            self.assertTrue(bool(refused.reason[index[0]] & driver.CELLS))
            self.assertFalse(bool(refused.scored[index[0]]))
            self.assertFalse(bool(refused.encoded.observation.attention_mask[index[0]].any()))


if __name__ == '__main__':
    unittest.main()
