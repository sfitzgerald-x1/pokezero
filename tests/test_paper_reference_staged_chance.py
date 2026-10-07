"""Guarded chance conditioning never omits policy mass or hides refusals."""
from copy import deepcopy
from fractions import Fraction
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from pokezero.mcts_eval.paper_reference import ReferenceRefusal, SamplingDeadlineExceeded
from pokezero.mcts_eval.paper_reference_staged_chance import (
    ConstantChancePlan, build_constant_chance_plan, validate_active_support, sample_staged_path,
)

MODULE = 'pokezero.mcts_eval.paper_reference_staged_chance'


def support_snapshot(stage=0):
    def mon(species, level, ability, hp, stats, types, moves):
        return dict(species='[Species:'+species+']',set=dict(level=level),ability=ability,
            item='leftovers',hp=hp,maxhp=hp,baseMaxhp=hp,storedStats=stats,types=types,
            speed=stats['spe'],boosts=dict(atk=0,def_=0,spa=0,spd=0,spe=0),
            isActive=True,status='',volatiles={},moveSlots=[dict(id=m,pp=15) for m in moves])
    own=mon('lugia',70,'pressure',264,dict(atk=131,def_=223,spa=167,spd=257,spe=195),
            ['Psychic','Flying'],['substitute','recover','toxic','psychic'])
    opp=mon('shuckle',98,'sturdy',198,dict(atk=75,def_=506,spa=75,spd=506,spe=65),
            ['Bug','Rock'],['wrap','rest','toxic','encore'])
    for row in (own,opp):
        row['storedStats']['def']=row['storedStats'].pop('def_')
    own['volatiles']=dict(encore=dict(move='psychic',duration=5-stage),substitute=dict(hp=66 if stage==0 else 61))
    if stage==2:
        opp.update(status='slp',statusState=dict(time=3,startTime=3,skippedTime=0,source='[Pokemon:p2a]'))
    return NS(bridge_snapshot=dict(battle=dict(formatid='gen3randombattle',gameType='singles',
        field=dict(weather='',terrain='',pseudoWeather=dict(sleepclausemod={})),sides=[
            dict(id='p1',pokemon=[own],sideConditions={}),
            dict(id='p2',pokemon=[opp],sideConditions=dict(spikes=dict(layers=2)))])))


class PlanTests(unittest.TestCase):
    def factory(self):
        own,opp='p1a: Lugia','p2a: Shuckle'
        chunks=[
            ('|move|'+own+'|Psychic|'+opp,'|move|'+opp+'|Wrap|'+own,
             '|-activate|'+own+'|Substitute|[damage]','|upkeep','|turn|13'),
            ('|move|'+own+'|Psychic|'+opp,'|move|'+opp+'|Rest|'+opp,
             '|-status|'+opp+'|slp|[from] move: Rest','|upkeep','|turn|14'),
            ('|move|'+own+'|Psychic|'+opp,'|cant|'+opp+'|slp','|upkeep','|turn|15'),
        ]
        before=NS(history=('opening',))
        state=NS(player_id='p1',history=before.history+sum(chunks,()))
        factory=NS(state=state,pending_transition=NS(own_action=3,continuation_actions=(3,3),before_state=before),
            set_source=NS(metadata=NS(source_hash='f5a5265143d423af'),universes=dict(shuckle=NS(variants=[
                NS(level=98,ability='Sturdy',item='Leftovers',moves=('rest','wrap','toxic','encore'))]))),
            env=NS(config=NS(resolved_showdown_root=lambda:Path('/pinned'))))
        return factory

    def build(self,factory):
        with mock.patch(MODULE+'.public_history',side_effect=lambda s:s.history), \
             mock.patch(MODULE+'.ENGINE_HASHES',{'engine':hashlib.sha256(b'pinned').hexdigest()}), \
             mock.patch.object(Path,'read_bytes',return_value=b'pinned'):
            return build_constant_chance_plan(factory)

    def test_public_program_not_a_seed_or_numeric_opponent_action_whitelist(self):
        plan=self.build(self.factory())
        self.assertEqual(plan.own_actions,(3,3,3))
        self.assertEqual(len(plan.expected_histories),3)
        self.assertNotIn('seed',plan.__dict__)

    def test_unsupported_public_program_retains_joint_dispatch(self):
        factory=self.factory()
        factory.pending_transition.continuation_actions=(3,None)
        self.assertIsNone(self.build(factory))
        factory=self.factory()
        factory.state.history=tuple(line.replace('|Wrap|','|Toxic|') for line in factory.state.history)
        self.assertIsNone(self.build(factory))

    def test_no_Encore_end_program_requires_deterministic_timer_survival(self):
        factory=self.factory()
        factory.state.history=(*factory.state.history[:-1],'|-end|p1a: Lugia|Encore',factory.state.history[-1])
        self.assertIsNone(self.build(factory))

    def test_engine_or_generator_support_drift_refuses_not_latent_rejection(self):
        factory=self.factory()
        factory.set_source.universes['shuckle'].variants[0].ability='Early Bird'
        with self.assertRaisesRegex(ReferenceRefusal,'generator support'):
            self.build(factory)
        with mock.patch(MODULE+'.public_history',side_effect=lambda s:s.history), \
             mock.patch(MODULE+'.ENGINE_HASHES',{'engine':'00'}), \
             mock.patch.object(Path,'read_bytes',return_value=b'pinned'), \
             self.assertRaisesRegex(ReferenceRefusal,'effective engine'):
            build_constant_chance_plan(self.factory())


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.plan=ConstantChancePlan('p1','p2',(3,3,3),(('one',),('two',),('three',)))

    def test_surviving_Encore_and_all_positive_later_Sub_HP(self):
        for stage in range(3):
            for duration in range(4-stage,7-stage):
                for hp in ((66,) if stage==0 else range(1,67)):
                    snapshot=support_snapshot(stage)
                    vol=snapshot.bridge_snapshot['battle']['sides'][0]['pokemon'][0]['volatiles']
                    vol['encore']['duration']=duration
                    vol['substitute']['hp']=hp
                    self.assertTrue(validate_active_support(self.plan,snapshot,stage))

    def test_only_deterministically_impossible_Encore_is_a_zero_likelihood_filter(self):
        snapshot=support_snapshot()
        own=snapshot.bridge_snapshot['battle']['sides'][0]['pokemon'][0]
        own['volatiles']['encore']['duration']=3
        self.assertFalse(validate_active_support(self.plan,snapshot,0))
        own['storedStats']['def']+=1
        with self.assertRaisesRegex(ReferenceRefusal,'damage/event support'):
            validate_active_support(self.plan,snapshot,0)

    def test_rest_source_and_sleep_clock_are_not_induced_sleep(self):
        snapshot=support_snapshot(2)
        opp=snapshot.bridge_snapshot['battle']['sides'][1]['pokemon'][0]
        opp['statusState']['source']='[Pokemon:p1a]'
        with self.assertRaisesRegex(ReferenceRefusal,'Rest-source'):
            validate_active_support(self.plan,snapshot,2)

    def test_unknown_benches_slot_order_and_opponent_PP_remain_in_full_policy(self):
        snapshot=support_snapshot(1)
        for side in snapshot.bridge_snapshot['battle']['sides']:
            side['pokemon'].append(dict(species='arbitrary hidden bench',ability='unknown'))
        opp=snapshot.bridge_snapshot['battle']['sides'][1]['pokemon'][0]
        opp['moveSlots'].reverse()
        opp['moveSlots'][0]['pp']=1
        self.assertTrue(validate_active_support(self.plan,snapshot,1))

    def test_all_four_sleep_blocked_moves_eligible_but_switch_mass_not_removed(self):
        for move in ('wrap','rest','toxic','encore'):
            self.assertTrue(self.plan.compatible_move(2,dict(kind='move',legal=True,move_id=move)))
        self.assertFalse(self.plan.compatible_move(2,dict(kind='switch',legal=True)))
        self.assertFalse(self.plan.compatible_move(0,dict(kind='move',legal=True,move_id='rest')))

    def test_timer_PP_field_and_fresh_Sub_mutations_refuse(self):
        edits=[lambda b:b['field'].update(weather='sandstorm'),
               lambda b:b['sides'][0]['pokemon'][0]['moveSlots'][3].update(pp=3),
               lambda b:b['sides'][0]['pokemon'][0]['volatiles']['substitute'].update(hp=65)]
        for edit in edits:
            snapshot=support_snapshot()
            edit(snapshot.bridge_snapshot['battle'])
            with self.assertRaises(ReferenceRefusal):
                validate_active_support(self.plan,snapshot,0)


class KernelTests(unittest.TestCase):
    def fixture(self):
        plan=ConstantChancePlan('p1','p2',(3,3,3),(('one',),('two',),('three',)))
        env=NS(stage=0,history=(),seed=None)
        env.snapshot=lambda:NS(stage=env.stage,history=env.history,bridge_snapshot=support_snapshot().bridge_snapshot)
        def restore(snapshot):
            env.stage,env.history=snapshot.stage,snapshot.history
        env.restore=restore
        env.reseed_simulator_rng=lambda seed:setattr(env,'seed',seed)
        def step(actions):
            env.history=plan.expected_histories[env.stage]
            if env.seed==10:
                env.history=('rejected chance',)
            env.stage+=1
        env.step=step
        env.requested_players=lambda:('p1','p2')
        env.observe=lambda player:NS(metadata=dict(action_candidates=[
            dict(action_index=0,kind='switch',legal=True),
            dict(action_index=1,kind='move',legal=True,move_id=('wrap','rest','toxic')[min(env.stage,2)])]))
        env.terminal=lambda:None
        env.public_materialization_state=lambda player:NS(history=env.history,deferred_opponent_action_player=None)
        root=NS(key=b'root')
        factory=NS(env=env,root=root,state=NS(history=('three',),deferred_opponent_action_player=None),
                   check_sampling_deadline=mock.Mock(),_release=lambda e:e.update(released=True))
        factory.evaluator=mock.Mock(return_value=((0,1),NS(priors=(.9,.1))))
        worlds=[]
        prior=mock.Mock()
        prior.receipts=[]
        prior.root=NS(key=b'anchor')
        def draw(rng):
            env.stage,env.history=0,()
            world=NS(closed=False,release=mock.Mock())
            world.close=lambda:(setattr(world,'closed',True),world.release())
            worlds.append(world)
            prior.receipts.append(dict(ordinal=len(worlds)-1,status='ROOT_VALIDATED',packed_team_sha256=str(len(worlds))))
            return world
        prior.side_effect=draw
        rng=NS(choices=mock.Mock(side_effect=[[0],[1],[1],[1]]),
               getrandbits=mock.Mock(side_effect=[10,11,12,13]))
        return factory,prior,plan,rng,worlds

    def test_full_policy_draw_once_per_stage_and_chance_retry_keeps_actual_private_state(self):
        factory,prior,plan,rng,worlds=self.fixture()
        evidence={}
        with mock.patch(MODULE+'.validate_active_support',return_value=True), \
             mock.patch(MODULE+'.public_history',side_effect=lambda s:s.history), \
             mock.patch('pokezero.mcts_eval.paper_reference_showdown.decision_state',return_value=factory.root):
            world=sample_staged_path(factory,prior,plan,rng,evidence)
        self.assertEqual(prior.call_count,2)
        self.assertEqual(rng.choices.call_count,4)
        for call in rng.choices.call_args_list:
            self.assertEqual(call.kwargs['weights'],(.9,.1))
            self.assertEqual(call.args[0],(0,1))
        self.assertTrue(worlds[0].closed)
        self.assertFalse(worlds[1].closed)
        witness=evidence['substitute_policy_conditioning']
        self.assertEqual([s['chance_attempts'] for s in witness['steps']],[2,1,1])
        self.assertEqual([s['chance_seed'] for s in witness['steps']],[11,12,13])
        self.assertEqual(witness['sampled_substitute_hp'],{'p1':66})
        world.close()
        self.assertTrue(evidence['released'])

    def test_deadline_closes_unfinished_world_and_cannot_accept_work(self):
        factory,prior,plan,rng,worlds=self.fixture()
        factory.check_sampling_deadline.side_effect=[None,SamplingDeadlineExceeded('clock')]
        evidence={}
        with mock.patch(MODULE+'.validate_active_support',return_value=True), \
             self.assertRaises(SamplingDeadlineExceeded):
            sample_staged_path(factory,prior,plan,rng,evidence)
        self.assertTrue(worlds[0].closed)
        self.assertNotIn('substitute_policy_conditioning',evidence)

    def test_support_mismatch_is_fatal_not_rejection_then_redraw(self):
        factory,prior,plan,rng,worlds=self.fixture()
        with mock.patch(MODULE+'.validate_active_support',side_effect=ReferenceRefusal('support')), \
             self.assertRaisesRegex(ReferenceRefusal,'support'):
            sample_staged_path(factory,prior,plan,rng,{})
        self.assertEqual(prior.call_count,1)
        self.assertTrue(worlds[0].closed)

    def test_exact_finite_posterior_with_variable_policy_and_private_chance(self):
        joint,staged={},{}
        for theta,p1 in ((0,Fraction(1,4)),(1,Fraction(3,4))):
            for private,p2 in ((0,Fraction(1,10)),(1,Fraction(9,10))):
                weight=Fraction(1,2)*p1*Fraction(1,2)*p2
                c=Fraction(1,8)*Fraction(1,16)*Fraction(1,256)
                joint[theta,private]=weight*c
                staged[theta,private]=weight
        def normalized(rows):
            return {k:v/sum(rows.values()) for k,v in rows.items()}
        self.assertEqual(normalized(joint),normalized(staged))


if __name__=='__main__':
    unittest.main()
