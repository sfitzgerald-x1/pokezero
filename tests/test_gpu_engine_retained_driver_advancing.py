"""Per-row driver advancement and one-pass timing, using fake models only."""
import unittest
import numpy as np
import jax
import jax.numpy as j

from gpu_engine import engine_retained_encoder_driver as d, transition
import test_gpu_engine_engine_own_facts as fixtures


class AdvancingDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DeviceOwnFactsTests.setUpClass()
        cls.f = fixtures.DeviceOwnFactsTests
        fake = lambda _: d.FakeScores(j.ones(9, j.float32), j.array(1, j.int32))
        cls.basic = d.prepare(cls.f.request, cls.f.compiled, fake_forward=fake)
        cls.full = d.prepare(cls.f.request, cls.f.compiled, fake_forward=fake, public_mode='retained-full')

    def construct(self, full=False, b=2):
        return d.construct(self.f.pool, np.tile([[0, 1]], (b, 1)), prepared=self.full if full else self.basic,
            stream_ids=j.arange(800, 800+b, dtype=j.int32), seed=37)

    def chance(self, b=2):
        draws = j.zeros((b, 2, 12), j.int32).at[:, :, 4].set(1)
        return transition.ChanceInputs(draws, *[j.zeros((b, 2), j.int32) for _ in range(4)], j.zeros(b, j.int32))

    def same(self, before, after, row=None):
        old, new = jax.tree.leaves(before), jax.tree.leaves(after)
        self.assertEqual(len(old), len(new))
        for a, b in zip(old, new):
            np.testing.assert_array_equal(a if row is None else a[row], b if row is None else b[row])

    def row_same(self, before, after, row):
        # Explicit driver schema helpers detach global namespaces; checking the
        # selected whole tree also checks those global fields byte-for-byte.
        mask = j.arange(before.state.core.frame_number.shape[0]) != row
        core = d._rows(mask, before.state.core._replace(originals=None, constructors=None),
            after.state.core._replace(originals=None, constructors=None))
        self.same(core, after.state.core._replace(originals=None, constructors=None))
        for a, b in zip(before.state.observers, after.state.observers):
            self.same(d._observer_rows(mask, a, b), b)
        for a, b in zip(before.owners, after.owners):
            self.same(d._report_rows(mask, a, b), b)

    def test_full_all_inactive_b8_retains_all_state_and_global_namespaces(self):
        session = self.construct(full=True, b=8)
        old = session.result
        original, constructors = session.state.core.originals, session.state.core.constructors
        result = d.advance(session, j.full((8, 2), 8, j.int32), active_rows=j.zeros(8, bool),
            chance_inputs=self.chance(8))
        self.same(old, result)
        self.assertIs(result.state.core.originals, original)
        self.assertIs(result.state.core.constructors, constructors)
        diag = d.latest_diagnostics(session)
        self.assertEqual(diag.active_rows.tolist(), [False]*8)
        self.assertEqual(diag.executed_rows.tolist(), [False]*8)
        self.assertFalse(bool(diag.invalid_actions.any()))
        self.assertFalse(bool(diag.placeholder_moves.any()))
        # Global [8] namespaces in refresh must never be mistaken for B rows.
        before = old.state.observers[0].retained.context.refresh.controller.visible
        after = result.state.observers[0].retained.context.refresh.controller.visible
        self.assertEqual(before.source_binding.shape, (8,))
        self.same(before.source_binding, after.source_binding)
        self.same(before.schema_binding, after.schema_binding)

    def test_masked_frame_has_no_callbacks_cursor_changes_or_diagnostics(self):
        session = self.construct()
        core = session.state.core
        frame = d._checked(self.basic)['frame'](core, j.full((2, 2), 8, j.int32), self.chance(), j.array([False, True]))
        self.same(core.cursor, frame.cursor, row=0)
        self.same(frame.before_registry, frame.after_registry, row=0)
        self.assertEqual(frame.callbacks.count[0], 0)
        self.assertFalse(bool(frame.callbacks.words[0].any()))
        self.assertFalse(bool(frame.callbacks.event_turn[0].any()))
        self.assertFalse(bool(frame.callbacks.switch_details[0].any()))
        self.assertFalse(bool(frame.invalid_actions[0]))
        self.assertEqual(frame.placeholder_moves[0], 0)
        self.assertGreater(int(frame.callbacks.count[1]), 0)

    def test_mixed_mask_profile_and_unprofiled_have_identical_world_once(self):
        a, b = self.construct(), self.construct()
        before = a.result
        names = []
        def timing(name, milliseconds):
            self.assertTrue(np.isfinite(milliseconds) and milliseconds >= 0)
            names.append(name)
        actions, mask = j.full((2, 2), 8, j.int32), j.array([False, True])
        expected = d.advance(a, actions, active_rows=mask, chance_inputs=self.chance())
        actual = d.advance(b, actions, active_rows=mask, timing_callback=timing, chance_inputs=self.chance())
        self.same(expected, actual)
        self.row_same(before, actual, 0)
        self.assertEqual(actual.state.core.frame_number.tolist(), [0, 1])
        self.assertEqual(names, ['engine', 'requests', 'public_p1', 'encode_p1', 'admission_p1',
            'public_p2', 'encode_p2', 'admission_p2'])
        self.assertEqual(d.latest_diagnostics(b).executed_rows.tolist(), [False, True])
        self.assertEqual(d.latest_diagnostics(b).active_rows.tolist(), [False, True])
        for report in actual.owners:
            self.assertFalse(bool(report.B4_qualified.any()))

    def test_prior_public_refusal_freezes_row_without_using_global_policy_gate(self):
        session = self.construct()
        # Explicit synthetic corruption of the internal operational admission
        # exercises the freeze branch without claiming a new scientific run.
        obs = session.state.observers[0]
        obs = obs._replace(projection=obs.projection._replace(admitted=j.array([False, True])))
        state = session.state._replace(observers=(obs, session.state.observers[1]))
        result = session.result._replace(state=state)
        session.state, session.result = state, result
        d._SESSIONS[session].update(state=state, result=result)
        out = d.advance(session, j.full((2, 2), 8, j.int32), chance_inputs=self.chance())
        self.row_same(result, out, 0)
        self.assertEqual(d.latest_diagnostics(session).active_rows.tolist(), [True, True])
        self.assertEqual(d.latest_diagnostics(session).executed_rows.tolist(), [False, True])
        self.assertEqual(out.state.core.frame_number.tolist(), [0, 1])
        self.assertFalse(bool(result.owners[1].policy_admitted.any()))

    def test_host_validation_and_callback_reentry_leave_current_carry_uncommitted(self):
        session = self.construct()
        old = session.state
        actions = j.full((2, 2), 8, j.int32)
        for kwargs in ({'active_rows': j.array([0, 1])}, {'active_rows': j.ones((2, 1), bool)},
                {'active_rows': [True, False]}, {'timing_callback': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                d.advance(session, actions, chance_inputs=self.chance(), **kwargs)
            self.assertIs(session.state, old)
        def reenter(*_):
            d.advance(session, actions, chance_inputs=self.chance())
        with self.assertRaisesRegex(ValueError, 'not reentrant'):
            d.advance(session, actions, timing_callback=reenter, chance_inputs=self.chance())
        self.assertIs(session.state, old)
        self.assertFalse(d._SESSIONS[session]['advancing'])
        with self.assertRaises(ValueError):
            d.latest_diagnostics(d.Session(session.prepared, session.state, session.result))
        d.advance(session, actions, active_rows=j.zeros(2, bool), chance_inputs=self.chance())
        self.assertFalse(bool(d.latest_diagnostics(session).executed_rows.any()))


if __name__ == '__main__':
    unittest.main()
