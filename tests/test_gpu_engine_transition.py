"""Turn-plumbing fixtures only; no game-fidelity or benchmark certification."""
import importlib.util
import json
import os
import unittest


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires isolated spike JAX")
class TransitionTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as jnp
        from gpu_engine import mechanics as m
        from gpu_engine import transition as t
        from gpu_engine.state import empty_state
        self.jax, self.jnp, self.m, self.t, self.empty = jax, jnp, m, t, empty_state
        self.table = t.MoveTable(
            jnp.array([0, 1000, 0, 50]), jnp.array([0, 1, 1, 0]),
            jnp.array([False, True, False, True]), jnp.array([100]*4),
            jnp.array([True]*4), jnp.zeros(4, jnp.int32),
            jnp.ones((18,18), jnp.int32), jnp.ones((18,18), jnp.int32), jnp.array(3))

    def live(self, batch=1):
        j = self.jnp
        s = self.empty(batch, seed=29)
        stats = j.full_like(s.stats, 200).at[..., 0].set(100)
        stats = stats.at[:,0,:,5].set(300)
        return s._replace(stats=stats, level=j.full_like(s.level,100),
            hp=j.full_like(s.hp,100), max_hp=j.full_like(s.hp,100),
            moves=j.ones_like(s.moves), pp=j.full_like(s.pp,10),
            types=s.types.at[...,0].set(1), done=j.zeros_like(s.done))

    def assert_state_equal(self, a, b):
        for name in a._fields:
            with self.subTest(field=name):
                self.assertTrue(bool(self.jnp.array_equal(getattr(a,name),getattr(b,name))))

    def test_faster_ko_suppresses_second_move_and_requires_replacement(self):
        s = self.live()
        r = self.t.step(s,self.jnp.zeros((1,2),self.jnp.int32),self.table)
        self.assertEqual(r.state.hp[0,:,0].tolist(),[100,0])
        self.assertEqual(r.state.pp[0,:,0,0].tolist(),[9,10])
        self.assertEqual(r.state.forced_switch.tolist(),[[False,True]])
        self.assertEqual(r.placeholder_moves.tolist(),[1])
        self.assertEqual(r.state.turn.tolist(),[1])

    def test_priority_can_override_speed(self):
        s = self.live()._replace(moves=self.live().moves.at[0,1,0,0].set(3))
        table = self.table._replace(priority=self.table.priority.at[3].set(1),
                                   power=self.table.power.at[3].set(1000))
        r = self.t.step(s,self.jnp.zeros((1,2),self.jnp.int32),table)
        self.assertEqual(r.state.hp[0,:,0].tolist(),[0,100])

    def test_unilateral_replacement_is_not_an_opponent_move_or_residual_turn(self):
        s = self.live()
        s = s._replace(hp=s.hp.at[0,0,0].set(0),
                       status=s.status.at[0,1,0].set(self.m.TOX))
        r = self.t.step(s,self.jnp.array([[4,0]]),self.table)
        self.assertEqual(r.state.active.tolist(),[[1,0]])
        self.assertEqual(r.state.hp[0,1,0].item(),100)
        self.assertEqual(r.state.pp[0,1,0,0].item(),10)
        self.assertEqual(r.state.status_counter[0,1,0].item(),0)
        self.assertEqual(r.state.turn.tolist(),[0])
        self.assertEqual(r.placeholder_moves.tolist(),[0])

    def test_invalid_and_done_battles_remain_byte_identical(self):
        s = self.live(2)._replace(done=self.jnp.array([False,True]))
        r = self.t.step(s,self.jnp.array([[9,0],[-100,100]]),self.table)
        self.assertEqual(r.invalid_actions.tolist(),[True,False])
        self.assert_state_equal(s,r.state)
        self.assertEqual(r.placeholder_moves.tolist(),[0,0])

    def test_sleep_blocks_pp_and_toxic_residual_ticks(self):
        s = self.live()
        s = s._replace(status=s.status.at[0,0,0].set(self.m.SLP).at[0,1,0].set(self.m.TOX),
            status_counter=s.status_counter.at[0,0,0].set(3).at[0,1,0].set(2),
            moves=s.moves.at[0,1,0,0].set(2))
        r = self.t.step(s,self.jnp.zeros((1,2),self.jnp.int32),self.table)
        self.assertEqual(r.state.pp[0,:,0,0].tolist(),[10,9])
        self.assertEqual(r.state.status_counter[0,:,0].tolist(),[2,3])
        self.assertEqual(r.state.hp[0,:,0].tolist(),[100,82])

    def test_switch_precedes_attack_and_clears_departing_boosts(self):
        s = self.live()._replace(boosts=self.live().boosts.at[0,0,0,0].set(6))
        r = self.t.step(s,self.jnp.array([[4,0]]),self.table)
        self.assertEqual(r.state.active.tolist(),[[1,0]])
        self.assertEqual(r.state.hp[0,0,:2].tolist(),[100,0])
        self.assertEqual(r.state.boosts[0,0,0,0].item(),0)

    def test_party_exhaustion_sets_winner(self):
        s = self.live()
        s = s._replace(hp=s.hp.at[0,1,1:].set(0))
        r = self.t.step(s,self.jnp.zeros((1,2),self.jnp.int32),self.table)
        self.assertEqual(r.state.done.tolist(),[True])
        self.assertEqual(r.state.winner.tolist(),[0])

    def test_last_mon_ko_ends_battle_before_lethal_status_residual(self):
        for status in (self.m.BRN,self.m.TOX):
            with self.subTest(status=status):
                s = self.live()
                s = s._replace(hp=s.hp.at[:,:,1:].set(0).at[0,0,0].set(1),
                    status=s.status.at[0,0,0].set(status),
                    status_counter=s.status_counter.at[0,0,0].set(2))
                r = self.t.step(s,self.jnp.zeros((1,2),self.jnp.int32),self.table)
                self.assertEqual(r.state.hp[0,:,0].tolist(),[1,0])
                self.assertEqual(r.state.status_counter[0,0,0].item(),2)
                self.assertEqual(r.state.done.tolist(),[True])
                self.assertEqual(r.state.winner.tolist(),[0])

    def test_jit_matches_eager_and_per_battle_stream_is_batch_invariant(self):
        j = self.jnp
        s = self.live(2)
        actions = j.zeros((2,2),j.int32)
        eager = self.t.step(s,actions,self.table)
        compiled = self.jax.jit(self.t.step)(s,actions,self.table)
        compiled.state.hp.block_until_ready()
        self.assert_state_equal(eager.state,compiled.state)
        one = self.t.step(self.live(),actions[:1],self.table)
        for name in s._fields:
            self.assertTrue(bool(j.array_equal(getattr(one.state,name),getattr(eager.state,name)[:1])))

    def test_rejects_wrong_action_shape_and_dtype(self):
        s = self.live()
        for actions in (self.jnp.zeros((1,)),self.jnp.zeros((1,2))):
            with self.assertRaises(ValueError): self.t.step(s,actions,self.table)

    def test_pool_initialization_rejects_invalid_selection(self):
        import numpy as np
        pool = {"species":np.ones((3,6),np.int32)}
        for selection in ([[0,3]], [[-1,0]], [[True,False]], [0,1], [[0.0,1.0]]):
            with self.assertRaises(ValueError): self.t.state_from_pool(pool,selection)

    @unittest.skipUnless(os.environ.get("POKEZERO_GPU_TEAM_POOL"), "requires real generated team pool")
    def test_real_team_initialization_and_one_compiled_turn(self):
        import numpy as np
        with np.load(os.environ["POKEZERO_GPU_TEAM_POOL"],allow_pickle=False) as pool:
            metadata = json.loads(pool["metadata_utf8"].tobytes())
            table = self.t.tables_from_metadata(metadata)
            s = self.t.state_from_pool(pool,np.arange(512).reshape(256,2),seed=91)
        actions = self.jnp.zeros((256,2),self.jnp.int32)
        r = self.jax.jit(self.t.step)(s,actions,table)
        r.state.hp.block_until_ready()
        self.assertFalse(bool(self.jnp.any(r.invalid_actions)))
        self.assertTrue(bool(self.jnp.all(r.state.hp >= 0)))
        self.assertTrue(bool(self.jnp.all(r.state.hp <= r.state.max_hp)))
        self.assertTrue(bool(self.jnp.all(r.state.turn == 1)))
        self.assertTrue(bool(self.jnp.all((r.placeholder_moves >= 1) & (r.placeholder_moves <= 2))))


if __name__ == "__main__":
    unittest.main()
