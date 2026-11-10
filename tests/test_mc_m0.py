"""Monte Carlo scheduling, censoring and paired diagnostics without a device."""
import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
from gpu_engine import mc_m0 as m0, mc_root_labeller as lab
from gpu_engine import mc_m0_readout as readout


class Env:
    def restore(self, snapshot):self.turn=0
    def reseed_simulator_rng(self, seed):self.rng=np.random.default_rng(seed)
    def requested_players(self):return ('p1','p2')
    def observe(self, player):return SimpleNamespace(legal_action_mask=[True, True]+[False]*7)
    def step(self, actions):self.turn+=1;self.winner='p1' if self.rng.random()<.5 else 'p2'
    def terminal(self):return SimpleNamespace(winner=self.winner,capped=False) if self.turn>=3 else None


class Model:
    def observe(self, env, player, seed):return env.observe(player)
    def probabilities(self, observations):return np.tile([.5,.5]+[0.]*7,(len(observations),1))


class MeasurementTests(unittest.TestCase):
    def test_nested_streams_are_identical_under_batched_scheduling(self):
        model=Model();root=dict(ours='p1',opp='p2',seed=42,root_seed=123,actions=[0,1],opp_replies=[0]*256)
        y,d,s=lab.run_root([Env()],dict(ours=model,opp=model),root,None,64,False)
        z,e,t=lab.run_root([Env() for _ in range(5)],dict(ours=model,opp=model),root,None,256,True)
        np.testing.assert_array_equal(y,z[:,:64]);np.testing.assert_array_equal(d,e[:,:64])
        self.assertEqual(s,[r[:64] for r in t]);self.assertTrue((d==6).all())

    def test_caps_are_unknown_and_draw_is_known(self):
        for terminal, expected in ((None,(None,'decision_cap')),
                (SimpleNamespace(capped=True,winner=None),(None,'simulator_cap')),
                (SimpleNamespace(capped=False,winner=None),(0.,'terminal'))):
            self.assertEqual(lab.outcome(SimpleNamespace(terminal=lambda:terminal),'p1'),expected)

    def test_censored_pair_has_bounds_and_cannot_be_separable(self):
        result=m0.pair_stats([[1,None,1],[0,-1,-1]], [['terminal','decision_cap','terminal'],['terminal']*3],3)[0]
        self.assertEqual(result['complete_pairs'],2);self.assertIsNone(result['separable'])
        self.assertEqual((result['a'],result['b']),(1,0))
        self.assertEqual(result['gap_identification_bounds'],[-5/6,-.5])
        with self.assertRaisesRegex(ValueError,'provenance'):
            m0.pair_stats([[-.25],[1]],[['decision_cap'],['terminal']],1)

    def test_negative_correlation_is_retained(self):
        r=m0.pair_stats([[1,-1]*32,[-1,1]*32],[['terminal']*64]*2,64)[0]
        self.assertAlmostEqual(r['rho'],-1);self.assertFalse(r['separable'])

    def test_sampler_rejects_bad_mass_and_never_returns_illegal_tail(self):
        legal=[True,True]+[False]*7
        self.assertEqual(lab.sample([.1,.9]+[0.]*7,legal,np.nextafter(1.,0.)),1)
        for p in ([0.]*9,[float('nan')]*9,[-1.]*9):
            with self.assertRaises(ValueError):lab.sample(p,legal,.1)

    def test_fixed_roster_rejects_duplicate_or_shrunken_scope_before_file_reads(self):
        roots=[dict(kind='self' if i<25 else 'foulplay',seed=i,target=1,root_seed=i,ours='p1') for i in range(50)]
        for changed in (roots[:-1], roots[:1]*25+roots[25:]):
            with self.assertRaises(ValueError):m0.validate(dict(roots=changed))

    def test_readout_does_not_turn_empty_or_declared_success_into_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            m0.write(root/'config.json',dict(roots=[{}]*50))
            m0.write(root/'manifest.json',dict(config_sha256=m0.sha(root/'config.json')))
            report=readout.readout(root)
            self.assertFalse(report['experiment_completion_verified'])
            self.assertEqual(report['completed_both_modes'],0)
            m0.write(root/'summary.json',dict(status='COMPLETE'))
            report=readout.readout(root)
            self.assertFalse(report['experiment_completion_verified'])
            (root/'config.json').write_text(json.dumps(dict(roots=[])))
            with self.assertRaisesRegex(ValueError,'changed configuration'):readout.readout(root)

    def test_failure_prevents_completion_even_with_complete_result_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            roots=[dict(kind='self' if i<25 else 'foulplay',seed=i,actions=[0,1]) for i in range(50)]
            m0.write(root/'config.json',dict(roots=roots))
            m0.write(root/'manifest.json',dict(config_sha256=m0.sha(root/'config.json')))
            m0.write(root/'prepared.json',[dict(index=i,status='READY',root=r) for i,r in enumerate(roots)])
            m0.write(root/'runtime-smoke.json',[dict(kind=k,status='PASS') for k in ('self','foulplay')])
            for i,r in enumerate(roots):
                for k in (64,256):
                    y=[[1.]*k,[-1.]*k];statuses=[['terminal']*k]*2
                    m0.write(root/f'root-{i}-{k}.json',dict(index=i,root=r,K=k,mode='batched' if k==256 else 'unbatched',
                        outcomes=y,statuses=statuses,pairs={str(n):m0.pair_stats(y,statuses,n) for n in (64,256) if n<=k},
                        cpu_seconds=1.,wall_seconds=1.))
            m0.write(root/'summary.json',dict(status='COMPLETE'))
            self.assertTrue(readout.readout(root)['experiment_completion_verified'])
            m0.write(root/'failure-summary.json',dict(status='FAILED'))
            report=readout.readout(root)
            self.assertFalse(report['experiment_completion_verified']);self.assertEqual(report['status'],'FAILED')
            (root/'failure-summary.json').write_text('{}')
            self.assertFalse(readout.readout(root)['experiment_completion_verified'])
