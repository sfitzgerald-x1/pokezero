"""Synthetic owner/public boundary fixtures; no engine/world/model execution."""
from types import SimpleNamespace as NS
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import device_public_worlds as plans, device_public_world_state as mapping
from gpu_engine import engine_device_damage_history as damage
from gpu_engine import engine_device_trackers as trackers
from gpu_engine.device_public_teacher import REPLY_HISTORY_APPROXIMATION
from test_gpu_engine_device_public_teacher import facts


def same(a, b):
    aa, ta = jax.tree.flatten(a); bb, tb = jax.tree.flatten(b)
    assert ta == tb
    for x, y in zip(aa, bb):
        np.testing.assert_array_equal(x, y)


class DamageHistoryWorldPrivacyTests(unittest.TestCase):
    def test_both_owners_detach_removes_only_real_peer_exact_history(self):
        for owner in (0, 1):
            context, _, _, tables, _ = facts(owner)
            foe = 1-owner
            old = context.public.damage_history
            history = old._replace(taken=j.arange(8, dtype=j.float32).reshape(1, 2, 2, 2)/100,
                                  dealt=j.arange(8, 16, dtype=j.float32).reshape(1, 2, 2, 2)/100)
            public = context.public._replace(damage_history=history)
            source = tables.source
            first = plans.detach_root(context._replace(public=public), source,
                row=0, owner=owner, enable_device_worlds=True)
            poisoned = history._replace(hp=history.hp.at[:, foe].set(991),
                maximum=history.maximum.at[:, foe].set(1009),
                taken=history.taken.at[:, foe, 1].set(97),
                dealt=history.dealt.at[:, owner, 1].set(91))
            second = plans.detach_root(context._replace(public=public._replace(damage_history=poisoned)),
                source, row=0, owner=owner, enable_device_worlds=True)
            same(first.public, second.public)
            before = trackers.readout(public, j.asarray([owner], j.int32))
            after = trackers.readout(first.public, j.asarray([owner], j.int32))
            same(before, after)
            got = first.public.damage_history
            np.testing.assert_array_equal(got.hp[:, foe], 0)
            np.testing.assert_array_equal(got.maximum[:, foe], 0)
            np.testing.assert_array_equal(got.hp[:, owner], history.hp[:, owner])
            np.testing.assert_array_equal(got.maximum[:, owner], history.maximum[:, owner])
            np.testing.assert_array_equal(got.taken[:, :, 0], history.taken[:, :, 0])
            np.testing.assert_array_equal(got.dealt[:, :, 0], history.dealt[:, :, 0])
            np.testing.assert_array_equal(got.taken[:, owner, 1], history.taken[:, owner, 1])
            np.testing.assert_array_equal(got.dealt[:, foe, 1], history.dealt[:, foe, 1])
            np.testing.assert_array_equal(got.taken[:, foe, 1], 0)
            np.testing.assert_array_equal(got.dealt[:, owner, 1], 0)
            self.assertFalse(first.reply_private_history_available)
            self.assertFalse(REPLY_HISTORY_APPROXIMATION['exact_known_zero'])

    def test_sampled_future_baseline_preserves_current_sums_and_legacy_none(self):
        for owner in (0, 1):
            context, root, _, _, _ = facts(owner)
            foe = 1-owner
            # Isolate rebase from the sampler/constructor: declared synthetic
            # current sample, not a read of the real opposing active HP.
            sample = context.state._replace(active=context.state.active.at[0, foe].set(3),
                hp=context.state.hp.at[0, foe, 3].set(219),
                max_hp=context.state.max_hp.at[0, foe, 3].set(311))
            h = root.public.damage_history
            h = h._replace(taken=h.taken.at[0, foe, 0, 0].set(.12),
                          dealt=h.dealt.at[0, foe, 1, 0].set(.17))
            public = root.public._replace(damage_history=h)
            root = NS(public=public, address=root.address)
            rebased = mapping._rebase(root, {0: 3}, 3, state=sample)
            out = rebased.damage_history
            self.assertEqual(int(out.hp[0, foe]), 219)
            self.assertEqual(int(out.maximum[0, foe]), 311)
            np.testing.assert_array_equal(out.hp[:, owner], h.hp[:, owner])
            np.testing.assert_array_equal(out.maximum[:, owner], h.maximum[:, owner])
            np.testing.assert_array_equal(out.taken, h.taken)
            np.testing.assert_array_equal(out.dealt, h.dealt)
            after = sample._replace(hp=sample.hp.at[0, foe, 3].set(202))
            @jax.jit
            def advance(history, state):
                return damage.observe(history, state, j.asarray([True]), trackers._percent)
            advanced = advance(out, after)
            np.testing.assert_allclose(advanced.taken[0, foe, 1].sum(), 17/311, atol=1e-7, rtol=0)
            np.testing.assert_allclose(advanced.taken[0, foe, 0].sum(), .12+.06, atol=1e-7, rtol=0)
            # Existing None carrier remains supported, without inventing a
            # fresh history from sampled values or changing public features.
            root.public = root.public._replace(damage_history=None)
            self.assertIsNone(mapping._rebase(root, {0: 3}, 3, state=sample).damage_history)


if __name__ == '__main__':
    unittest.main()
