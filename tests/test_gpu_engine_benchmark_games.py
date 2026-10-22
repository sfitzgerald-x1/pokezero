"""Finite-cohort counting and request-driver checks, not a GPU measurement."""
import importlib.util
import unittest
from typing import NamedTuple


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated spike JAX')
class AdvancingBenchmarkTests(unittest.TestCase):
    def test_completions_are_not_request_turn_rate_projection(self):
        from gpu_engine.benchmark_games import cohort_summary
        s=cohort_summary([True,False,True],[20,90,40],seconds=10,requests=100,live_requests=190,ordinary_turns=150)
        self.assertEqual((s['completed_games'],s['censored_games'],s['games_per_second']),(2,1,.2))
        self.assertEqual((s['mean_completed_game_turns'],s['ordinary_game_turns_per_second']),(30.,15.))
        self.assertEqual(s['final_turns'],[20,90,40]);self.assertEqual(s['completed_mask'],[True,False,True])

    def test_zero_completions_remain_zero_with_no_completed_turn_mean(self):
        from gpu_engine.benchmark_games import cohort_summary
        s=cohort_summary([False,False],[50,50],seconds=1,requests=50,live_requests=100,ordinary_turns=100)
        self.assertEqual(s['games_per_second'],0);self.assertIsNone(s['mean_completed_game_turns'])
        for kwargs in ({'seconds':0,'requests':1},{'seconds':1,'requests':0}):
            with self.assertRaises(ValueError):cohort_summary([False],[1],live_requests=1,ordinary_turns=1,**kwargs)

    def test_render_retains_censoring_and_does_not_project_games_from_turns(self):
        from gpu_engine.benchmark_games import cohort_summary,render_markdown
        row=cohort_summary([False],[100],seconds=2,requests=100,live_requests=100,ordinary_turns=100)
        row.update(batch=1,status='REQUEST_CAP',engine={'mean_ms':.1},encode={'mean_ms':.2},forward={'mean_ms':.3})
        result={'device':'authored fixture','jax':'test','protocol_completed':True,'rows':[row]}
        text=render_markdown(result)
        self.assertIn('| 1 | REQUEST_CAP | 0.100 | 0.200 | 0.300 | 20.000 | 0 | 1 | 0.000 | — |',text)
        self.assertIn('Unfinished games remain censored',text)
        row.update(status='INVALID_ACTION_OR_NONFINITE',games_per_second=None,mean_end_to_end_request_ms=None)
        result['protocol_completed']=False;text=render_markdown(result)
        self.assertIn('Protocol completed: False',text)
        self.assertIn('INVALID_ACTION_OR_NONFINITE',text)
        self.assertIn('| invalid | 0 | 1 | invalid |',text)

    def test_masked_sampling_no_legal_and_stable_prefix(self):
        import jax
        import jax.numpy as j
        from gpu_engine.benchmark_games import sample_actions
        keys=jax.random.split(jax.random.PRNGKey(7),6).reshape((3,2,2))
        logits=j.zeros((3,2,9));legal=j.zeros_like(logits,dtype=bool).at[:,:,6].set(True).at[2].set(False)
        a=jax.jit(sample_actions)(logits,legal,keys)
        self.assertEqual(a.tolist(),[[6,6],[6,6],[-1,-1]])
        self.assertTrue(bool(j.array_equal(a[:1],sample_actions(logits[:1],legal[:1],keys[:1]))))

    def test_injected_forward_drives_valid_requests_and_keeps_done_identity(self):
        import jax
        import jax.numpy as j
        from gpu_engine.benchmark_games import driver
        from gpu_engine.transition import MoveTable
        from gpu_engine.state import empty_state
        class Heads(NamedTuple):
            policy_logits:object
            value:object
            opponent_action_logits:object
        table=MoveTable(j.array([0,35,50]),j.array([0,1,0]),j.ones(3,bool),j.full(3,100),
            j.ones(3,bool),j.zeros(3,j.int32),j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(2))
        s=empty_state(2,seed=7);s=s._replace(hp=s.hp.at[:,:,0].set(100),max_hp=j.full_like(s.hp,100),
            species=j.ones_like(s.species),level=j.full_like(s.level,100),stats=j.full_like(s.stats,100),
            moves=j.ones_like(s.moves),pp=j.full_like(s.pp,8),done=j.array([False,True]))
        forward=lambda o:Heads(j.zeros((2,2,9)),j.zeros((2,2)),j.zeros((2,2,9)))
        out,heads,actions,finite=driver(forward,table)(s,j.array(0),jax.random.split(jax.random.PRNGKey(9),4).reshape((2,2,2)))
        self.assertTrue(bool(finite));self.assertFalse(bool(j.any(out.invalid_actions)))
        self.assertEqual(actions[1].tolist(),[-1,-1]);self.assertEqual(int(out.state.turn[0]),1)
        for before,after in zip(s,out.state):self.assertTrue(bool(j.array_equal(before[1],after[1])))
