"""Trackers.public_order: Showdown's request-order rule in public-slot coordinates.

Synthetic hook transactions only: no engine turn, model, fold or game.
"""
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_device_public_hooks as h, engine_device_trackers as tr
from gpu_engine.state import empty_state


def state(b=2):
    s = empty_state(b);shape = (b, 2, 6)
    return s._replace(species=j.broadcast_to(j.arange(1, 13, dtype=j.int32).reshape(1, 2, 6), shape),
        level=j.full(shape, 80, j.int32), gender=j.ones(shape, j.int32), hp=j.full(shape, 200, j.int32),
        max_hp=j.full(shape, 200, j.int32), done=j.zeros(b, bool))


def reference(order, log):
    """Pure numpy replay: each used install moves its public slot to position 0."""
    order = np.array(order, copy=True)
    for row, side, slot in log:
        current = list(order[row, side])
        at = current.index(slot)
        current[at], current[0] = current[0], slot
        order[row, side] = current
    return order


def put_active(s, side, physical):
    return s._replace(active=s.active.at[:, side].set(j.asarray(physical, j.int32)))


class PublicOrderTests(unittest.TestCase):
    def test_schema_initialize_identity_and_no_readout(self):
        self.assertEqual(tr.SCHEMA, 'pokezero.engine_device_public_trackers.v3')
        self.assertEqual(tr.Trackers._fields[-1], 'public_order')
        s = put_active(state(), 1, j.array([3, 0], j.int32))
        for t in (tr.initialize(s), h.initialize(s).trackers):
            self.assertEqual(t.public_order.dtype, j.int32)
            np.testing.assert_array_equal(t.public_order, np.broadcast_to(np.arange(6), (2, 2, 6)))
            # The lead holds public slot 0 whatever its physical index.
            self.assertEqual(t.active_slot.tolist(), [[0, 0], [0, 0]])
        # Not encoded: the public view has no such field.
        self.assertNotIn('public_order', tr.PublicView._fields)

    def test_installs_0_1_2_0_give_0_2_1(self):
        s = state();c = h.initialize(s);yes = j.ones(2, bool)
        for physical in (3, 4, 0):                    # public slots 1, 2, then the lead's 0
            s = put_active(s, 1, physical);h.install(c, s, 1, yes)
        self.assertEqual(c.trackers.public_order[0, 1].tolist(), [0, 2, 1, 3, 4, 5])
        self.assertEqual(c.trackers.public_order[0, 0].tolist(), list(range(6)))
        # The physical request order took the same swaps.
        self.assertEqual(c.trackers.own_order[0, 1].tolist(), [0, 1, 2, 4, 3, 5])
        # Today's encounter-slot rebase rule would instead give the identity.
        self.assertEqual(c.trackers.physical_to_public[0, 1].tolist(), [0, -1, -1, 1, 2, -1])

    def test_unrevealed_slots_never_move(self):
        rng = np.random.default_rng(7);s = state(4);c = h.initialize(s);yes = j.ones(4, bool)
        revealed = [0, 2, 5]                          # three physical members ever installed
        for _ in range(12):
            s = put_active(s, 0, rng.choice(revealed, 4).astype(np.int32));h.install(c, s, 0, yes)
        order = np.asarray(c.trackers.public_order[:, 0])
        for row in range(4):
            self.assertEqual(sorted(order[row]), list(range(6)))
            self.assertEqual(order[row, 3:].tolist(), [3, 4, 5])          # public slots 3..5 unrevealed
            self.assertEqual(int(c.trackers.next_slot[row, 0]), 3)

    def test_drag_baton_and_replacement_in_one_step_follow_hook_call_order(self):
        s = state();c = h.initialize(s);yes = j.ones(2, bool)
        for physical in (1, 2):                       # reveal side 1's slots 1 and 2 earlier
            s = put_active(s, 1, physical);h.install(c, s, 1, yes)
        s = put_active(s, 1, 0);h.install(c, s, 1, yes)

        @jax.jit
        def one_step(trackers, before):
            ctx = h.Context(trackers)
            dragged = put_active(before, 1, j.full(2, 2, j.int32))                 # Roar drags in slot 2
            h.drag(ctx, before, dragged, 1, None, j.ones(2, bool))
            passed = put_active(dragged, 0, j.full(2, 4, j.int32))                 # Baton target, side 0
            h.install(ctx, passed, 0, j.ones(2, bool))
            replaced = put_active(passed, 1, j.full(2, 1, j.int32))                # forced replacement, slot 1
            h.install(ctx, replaced, 1, j.ones(2, bool))
            return ctx.trackers

        before_order = np.asarray(c.trackers.public_order)
        out = one_step(c.trackers, s)
        expected = reference(before_order, [(r, 1, 2) for r in range(2)] + [(r, 0, 1) for r in range(2)]
                             + [(r, 1, 1) for r in range(2)])
        np.testing.assert_array_equal(out.public_order, expected)
        self.assertEqual(out.public_order[0, 1].tolist(), [1, 0, 2, 3, 4, 5])   # [0,2,1] -> drag 2 -> [2,0,1] -> 1
        self.assertEqual(out.public_order[0, 0].tolist(), [1, 0, 2, 3, 4, 5])

    def test_inactive_rows_and_initial_installs_are_unchanged(self):
        s = state(3);c = h.initialize(s);before = c.trackers
        s = put_active(s, 1, 2);h.install(c, s, 1, j.array([True, False, True]))
        self.assertEqual(c.trackers.public_order[:, 1].tolist(), [[1, 0, 2, 3, 4, 5], list(range(6)), [1, 0, 2, 3, 4, 5]])
        for name in tr.Trackers._fields:
            x = getattr(before, name)
            if x is None:continue
            for a, b in zip(jax.tree.leaves(x), jax.tree.leaves(getattr(c.trackers, name))):
                np.testing.assert_array_equal(np.asarray(a)[1], np.asarray(b)[1], err_msg=name)
        # An initial (start-of-battle) install never reorders.
        fresh = h.initialize(s);order = fresh.trackers.public_order
        h.install(fresh, s, 1, j.ones(3, bool), initial=True)
        np.testing.assert_array_equal(fresh.trackers.public_order, order)

    def test_random_install_log_agrees_with_reference_replays(self):
        from gpu_engine.v2e_regress import replay_public_order
        rng = np.random.default_rng(1729);b = 6;s = state(b);c = h.initialize(s)
        pre = np.asarray(c.trackers.public_order);log = [];calls = []
        install = jax.jit(lambda trackers, st, side, use: h.install(h.Context(trackers), st, side, use).trackers,
                          static_argnums=2)
        for _ in range(40):
            side = int(rng.integers(0, 2));physical = rng.integers(0, 6, b).astype(np.int32);use = rng.random(b) < .7
            s = put_active(s, side, physical)
            c.trackers = install(c.trackers, s, side, j.asarray(use))
            slots = np.asarray(c.trackers.active_slot[:, side])
            log += [(r, side, int(slots[r])) for r in range(b) if use[r]]
            calls.append((np.full(b, side), slots, use))
        np.testing.assert_array_equal(c.trackers.public_order, reference(pre, log))
        np.testing.assert_array_equal(c.trackers.public_order, replay_public_order(pre, calls))
        for row in np.asarray(c.trackers.public_order).reshape(-1, 6):self.assertEqual(sorted(row), list(range(6)))

    def test_legacy_update_applies_the_rule_once_per_net_install(self):
        s = state(2);t = tr.initialize(s);a = j.zeros((2, 2), j.int32)
        for physical in (3, 4, 4, 0):                 # the repeated 4 is no net install
            after = put_active(s, 1, physical);t = tr.update(t, s, after, a, active_rows=j.array([True, False]));s = after
        self.assertEqual(t.public_order[:, 1].tolist(), [[0, 2, 1, 3, 4, 5], list(range(6))])

    def test_live_root_carry_without_the_field_is_seeded_identity_never_redacted(self):
        from gpu_engine.device_public_worlds import _redact_peer_private
        s = state(1);c = h.initialize(s)
        s = put_active(s, 1, 3);h.install(c, s, 1, j.ones(1, bool))
        kept = _redact_peer_private(c.trackers, 0)
        self.assertEqual(kept.public_order[0, 1].tolist(), [1, 0, 2, 3, 4, 5])     # public: not redacted
        self.assertEqual(kept.own_order[0, 1].tolist(), [-1] * 6)                  # the peer's physical order is
        seeded = _redact_peer_private(c.trackers._replace(public_order=None), 0)
        np.testing.assert_array_equal(seeded.public_order, np.broadcast_to(np.arange(6), (1, 2, 6)))
        # A carry without the field still steps: the install hook leaves it absent.
        bare = h.Context(c.trackers._replace(public_order=None));h.install(bare, put_active(s, 1, 0), 1, j.ones(1, bool))
        self.assertIsNone(bare.trackers.public_order)


if __name__ == '__main__':
    unittest.main()
