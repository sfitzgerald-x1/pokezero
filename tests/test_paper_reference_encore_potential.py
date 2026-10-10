import copy
from fractions import Fraction as F
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from _showdown_root import requires_showdown, showdown_root
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownEnv, LocalShowdownConfig
from pokezero.showdown_fixture import FixturePokemon, pack_team
from pokezero.mcts_eval.paper_reference import ReferenceRefusal
from pokezero.mcts_eval.paper_reference_pending import public_history
from pokezero.mcts_eval.paper_reference_encore_potential import necessary_own_encore_potential
from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory


class EncorePotentialTests(unittest.TestCase):
    def state(self,encore=True,force=False,deferred=None):
        return NS(history=('|turn|3',),replay=NS(volatiles={'p1':('encore',) if encore else ()}),
            self_request={'forceSwitch':[True]} if force else {},deferred_opponent_action_player=deferred)

    def constraint(self):
        return dict(static_details_disabled_ambiguity=False,own_encore=dict(side='p1',
            ident='p1a: Lugia',required_remaining=3,actor_known_pp=10,maximum_pp_payments=4,
            observed_residuals_before_end=2))

    def evaluate(self,env,state,certificate=None):
        with patch('pokezero.mcts_eval.paper_reference_encore_potential.public_history',lambda s:s.history), \
             patch('pokezero.mcts_eval.paper_reference_encore_potential.prepare_constraints',
                   return_value=certificate or self.constraint()):
            return necessary_own_encore_potential(env,state,state.history+('|turn|4',),'p1')

    def test_guarded_indicator_reads_only_own_hypothetical_timer(self):
        for duration in (2,3,4):
            shell=dict(battle={'sides':[{'id':'p1','pokemon':[
                {'isActive':True,'volatiles':{'encore':{'duration':duration}}}]},
                {'id':'p2','pokemon':'MUST NOT BE INSPECTED'}]})
            original=copy.deepcopy(shell)
            env=NS(_search_snapshot_permitted=True,snapshot=lambda:NS(bridge_snapshot=shell))
            accepted,evidence=self.evaluate(env,self.state())
            self.assertEqual(accepted,duration==3)
            self.assertEqual(evidence['indicator'],int(duration==3))
            self.assertEqual(shell,original)
            self.assertFalse(evidence['native_outcome_modified'])

    def test_ambiguous_or_unproven_constraints_do_not_read_native_timer(self):
        env=NS(_search_snapshot_permitted=True,snapshot=lambda: self.fail('must not read timer'))
        for state in (self.state(encore=False),self.state(force=True),self.state(deferred='p2')):
            accepted,evidence=self.evaluate(env,state)
            self.assertTrue(accepted); self.assertFalse(evidence['applicable'])
        for c in (dict(static_details_disabled_ambiguity=True,own_encore=None),
                  dict(static_details_disabled_ambiguity=False,own_encore=None)):
            accepted,evidence=self.evaluate(env,self.state(),c)
            self.assertTrue(accepted);self.assertFalse(evidence['applicable'])

    def test_live_hidden_world_and_nonprefix_inputs_refuse(self):
        with self.assertRaisesRegex(ReferenceRefusal,'owned hypothetical'):
            self.evaluate(NS(_search_snapshot_permitted=False),self.state())
        with patch('pokezero.mcts_eval.paper_reference_encore_potential.public_history',lambda s:s.history):
            with self.assertRaisesRegex(ReferenceRefusal,'exact observed public prefix'):
                necessary_own_encore_potential(NS(_search_snapshot_permitted=True),self.state(),(),'p1')

    def test_native_error_is_not_a_zero_likelihood(self):
        def broken(): raise RuntimeError('native transport failed')
        with self.assertRaisesRegex(RuntimeError,'native transport failed'):
            self.evaluate(NS(_search_snapshot_permitted=True,snapshot=broken),self.state())
        for timer in (None,True,0,-1):
            env=NS(_search_snapshot_permitted=True,snapshot=lambda:NS(bridge_snapshot={'battle':{'sides':[
                {'id':'p1','pokemon':[{'isActive':True,'volatiles':{'encore':{'duration':timer}}}]}]}}))
            with self.assertRaisesRegex(ReferenceRefusal,'timer missing or invalid'):self.evaluate(env,self.state())

    def test_composed_guided_weights_and_redundant_indicators_recover_full_likelihood(self):
        # A rare hint-miss proposal gets high importance weight. Moving the
        # NECESSARY indicator earlier must kill it, not drop its correction.
        target={(team,d):F(1,8) for team in (0,1) for d in (3,4,5,6)}
        q={3:F(37,40),4:F(1,40),5:F(1,40),6:F(1,40)}
        for stages in (1,2,5):
            late={};early={}
            for key,p in target.items():
                team,d=key; final_likelihood=F(team+1,3)*int(d==3)
                joint_q=F(1,2)*q[d]; correction=p/joint_q
                late[key]=joint_q*correction*final_likelihood
                early[key]=joint_q*correction*int(d==3)**stages*final_likelihood
            self.assertEqual(late,early)
            self.assertEqual(sum(early.values()),F(1,8))

    def test_opt_in_requires_particles_and_public_anchor_constraints(self):
        self.assertFalse(ShowdownWorkerFactory('a','b','c','d').early_encore_potential)
        for kwargs in ({'early_encore_potential':True},{'history_particles':32,'early_encore_potential':True},
                       {'early_encore_potential':1}):
            with self.assertRaisesRegex(ReferenceRefusal,'early Encore potential'):
                ShowdownWorkerFactory('a','b','c','d',**kwargs)
        self.assertTrue(ShowdownWorkerFactory('a','b','c','d',history_particles=32,
            public_anchor_constraints=True,early_encore_potential=True).early_encore_potential)


@requires_showdown()
class NativeEncorePotentialTests(unittest.TestCase):
    def test_native_guard_retains_actual_timer_rejects_wrong_hypothetical_and_does_not_mutate(self):
        override=BattleStartOverride(observation_format_id='gen3randombattle',player_teams={
            'p1':pack_team((FixturePokemon('Aipom',('Growl','Tackle'),ability='Run Away'),)),
            'p2':pack_team((FixturePokemon('Wobbuffet',('Encore','Splash'),ability='Shadow Tag'),))})
        with LocalShowdownEnv(LocalShowdownConfig(showdown_root=showdown_root(),set_belief_source=True)) as env:
            env.reset_with_start_override(seed=223,start_override=override)
            env.step({'p1':0,'p2':0})
            start=env.snapshot(); current=env.public_materialization_state('p1')
            duration=start.bridge_snapshot['battle']['sides'][0]['pokemon'][0]['volatiles']['encore']['duration']
            for _ in range(duration): env.step({'p1':0,'p2':1})
            expected=public_history(env.public_materialization_state('p1'))
            env.restore(start)
            before=copy.deepcopy(env.snapshot().bridge_snapshot)
            accepted,evidence=necessary_own_encore_potential(env,current,expected,'p1')
            self.assertTrue(accepted);self.assertTrue(evidence['applicable'])
            self.assertEqual(evidence['certificate']['required_remaining'],duration)
            self.assertEqual(env.snapshot().bridge_snapshot,before)
            wrong=copy.deepcopy(start)
            wrong.bridge_snapshot['battle']['sides'][0]['pokemon'][0]['volatiles']['encore']['duration']=duration+1
            env.restore(wrong)
            accepted,evidence=necessary_own_encore_potential(env,current,expected,'p1')
            self.assertFalse(accepted);self.assertEqual(evidence['hypothetical_native_remaining'],duration+1)


if __name__=='__main__':unittest.main()
