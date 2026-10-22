"""Expansion shape/branch semantics, not a GPU throughput measurement."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated spike JAX')
class ExpansionTests(unittest.TestCase):
    def test_all_81_lexicographic_pairs_per_parent(self):
        from gpu_engine.benchmark_expansion import joint_actions
        a=joint_actions(2)
        self.assertEqual(a.shape,(162,2))
        self.assertEqual(a[:81].tolist(),[[x,y] for x in range(9) for y in range(9)])
        self.assertEqual(a[:81].tolist(),a[81:].tolist())
        with self.assertRaises(ValueError):joint_actions(0)

    def test_expand_matches_direct_request_and_freezes_invalid_children(self):
        import jax
        import jax.numpy as j
        from gpu_engine.benchmark_expansion import expand
        from gpu_engine.transition import MoveTable,step
        from gpu_engine.state import empty_state
        table=MoveTable(j.array([0,35,50]),j.array([0,1,0]),j.ones(3,bool),j.full(3,100),j.ones(3,bool),
            j.zeros(3,j.int32),j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(2))
        s=empty_state(2,seed=7);s=s._replace(hp=s.hp.at[:,:,0].set(100),max_hp=j.full_like(s.hp,100),
            level=j.full_like(s.level,100),stats=j.full_like(s.stats,100),moves=j.ones_like(s.moves),
            pp=j.full_like(s.pp,8),done=j.zeros_like(s.done))
        result=jax.jit(expand)(s,table);direct=step(s,j.zeros((2,2),j.int32),table)
        self.assertEqual(result.state.hp.shape,(162,2,6))
        self.assertEqual(int(j.sum(~result.invalid_actions)),32)
        for old,new,expected in zip(s,result.state,direct.state):
            self.assertTrue(bool(j.array_equal(new[0],expected[0])))
            self.assertTrue(bool(j.array_equal(new[81],expected[1])))
            self.assertTrue(bool(j.array_equal(new[80],old[0])))
        small=jax.jit(expand)(jax.tree.map(lambda x:x[:1],s),table)
        for a,b in zip(result.state,small.state):self.assertTrue(bool(j.array_equal(a[:81],b)))

    def test_markdown_separates_valid_transitions_and_invalid_slots(self):
        from gpu_engine.benchmark_expansion import render_markdown
        text=render_markdown({'device':'authored fixture','jax':'test','rows':[dict(nodes=1,mean_ms=2,
            node_expansions_per_second=500,child_slots_per_second=40500,valid_child_transitions=16,
            valid_child_transitions_per_second=8000)]})
        self.assertIn('| 1 | 2.000 | 500.0 | 40500.0 | 16 | 8000.0 |',text)
        self.assertIn('illegal requests return frozen states',text)
