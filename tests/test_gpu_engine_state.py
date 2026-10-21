"""Focused device-capable tests; no model, game or collector invocation."""
import importlib.util
import unittest

JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(JAX_AVAILABLE, "requires isolated spike JAX environment")
class GPUStateTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as jnp
        from gpu_engine.state import empty_state, legal_actions
        self.jax, self.jnp, self.empty, self.legal = jax, jnp, empty_state, legal_actions

    def live(self, batch=2):
        s = self.empty(batch, seed=17)
        return s._replace(hp=self.jnp.ones_like(s.hp)*100,
                          pp=self.jnp.ones_like(s.pp)*10, done=self.jnp.zeros_like(s.done))

    def test_arrays_have_leading_batch_and_static_layout(self):
        s = self.empty(7)
        self.assertEqual(s.stats.shape, (7, 2, 6, 6))
        self.assertEqual(s.moves.shape, (7, 2, 6, 4))
        self.assertEqual(s.boosts.shape, (7, 2, 6, 7))
        self.assertTrue(all(x.shape[0] == 7 for x in self.jax.tree.leaves(s)))
        self.assertFalse(bool(self.jnp.any(self.legal(s))))

    def test_invalid_sizes_and_seed_are_rejected(self):
        for b in (0, -1, True, 1.5):
            with self.subTest(b=b), self.assertRaises(ValueError): self.empty(b)
        for seed in (-1, 2**32, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError): self.empty(1, seed=seed)

    def test_keys_do_not_change_with_batch_size(self):
        self.assertTrue(bool(self.jnp.all(self.empty(2, seed=9).keys == self.empty(5, seed=9).keys[:2])))
        self.assertFalse(bool(self.jnp.all(self.empty(2, seed=9).keys == self.empty(2, seed=10).keys)))

    def test_active_slot_is_omitted_from_switch_actions(self):
        from gpu_engine.state import bench_indices
        active = self.jnp.array([[0, 5], [2, 3]], self.jnp.int32)
        self.assertEqual(bench_indices(active).tolist(), [[[1,2,3,4,5],[0,1,2,3,4]],[[0,1,3,4,5],[0,1,2,4,5]]])

    def test_pp_disabled_fainted_bench_and_trapping(self):
        s = self.live()
        s = s._replace(pp=s.pp.at[0,0,0,1].set(0), disabled=s.disabled.at[0,0,0,2].set(True),
                       hp=s.hp.at[0,0,3].set(0), trapped=s.trapped.at[1,1].set(True))
        mask = self.legal(s)
        self.assertEqual(mask.shape, (2,2,9))
        self.assertEqual(mask[0,0].tolist(), [True,False,False,True,True,True,False,True,True])
        self.assertFalse(bool(self.jnp.any(mask[1,1,4:])))

    def test_forced_replacement_overrides_trap_and_disables_moves(self):
        s = self.live(1)
        s = s._replace(hp=s.hp.at[0,0,0].set(0), trapped=s.trapped.at[0,0].set(True))
        self.assertEqual(self.legal(s)[0,0].tolist(), [False]*4+[True]*5)

    def test_struggle_and_compiled_mask(self):
        s = self.live(1)._replace(pp=self.jnp.zeros((1,2,6,4), self.jnp.int32))
        expected = self.legal(s)
        compiled = self.jax.jit(self.legal)(s)
        compiled.block_until_ready()
        self.assertEqual(expected[0,0,:4].tolist(), [True,False,False,False])
        self.assertTrue(bool(self.jnp.all(expected == compiled)))

    def test_living_active_forced_switch_cannot_move(self):
        s = self.live(1)
        s = s._replace(forced_switch=s.forced_switch.at[0,1].set(True),
                       trapped=s.trapped.at[0,1].set(True))
        self.assertEqual(self.legal(s)[0,1].tolist(), [False]*4+[True]*5)

    def test_nonzero_active_gathers_that_members_moves(self):
        s = self.live(1)
        s = s._replace(active=s.active.at[0,0].set(4),
                       pp=s.pp.at[0,0,4,1].set(0),
                       disabled=s.disabled.at[0,0,4,3].set(True))
        self.assertEqual(self.legal(s)[0,0,:4].tolist(), [True,False,True,False])


if __name__ == "__main__":
    unittest.main()
