"""Public structure, latent support and full-policy law of the ten-turn gate."""
from fractions import Fraction
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from pokezero.mcts_eval.paper_reference import ReferenceRefusal
from pokezero.mcts_eval.paper_reference_staged_chance import (
    StagedPrefixJointPlan, validate_active_support,
)
from pokezero.mcts_eval.paper_reference_wake_wrap_chance import (
    WakeWrapChancePlan, WAKE_WRAP_SCHEMA, WAKE_WRAP_JOINT_SCHEMA, build_wake_wrap_plan,
)
from test_paper_reference_staged_chance import PlanTests, support_snapshot

MODULE = 'pokezero.mcts_eval.paper_reference_wake_wrap_chance'


def ten_turn_factory():
    factory = PlanTests().rest_tail_factory()
    factory.pending_transition.continuation_actions = (3,3,3,3,0,0,0,0,0)
    own, opp = 'p1a: Lugia', 'p2a: Shuckle'
    fail = ('|move|'+own+'|Substitute|'+own, '|-fail|'+own+'|move: Substitute')
    chunks = [
        ('|move|'+own+'|Psychic|'+opp, '|-curestatus|'+opp+'|slp|[msg]',
            '|move|'+opp+'|Rest|'+opp, '|-status|'+opp+'|slp|[from] move: Rest',
            '|-heal|'+own+'|246/264|[from] item: Leftovers', '|-end|'+own+'|Encore', '|upkeep','|turn|17'),
        fail+('|cant|'+opp+'|slp', '|-heal|'+own+'|262/264|[from] item: Leftovers','|upkeep','|turn|18'),
        fail+('|cant|'+opp+'|slp', '|-heal|'+own+'|264/264|[from] item: Leftovers','|upkeep','|turn|19'),
        fail+('|-curestatus|'+opp+'|slp|[msg]', '|move|'+opp+'|Wrap|'+own,
            '|-activate|'+own+'|Substitute|[damage]','|upkeep','|turn|20'),
        fail+('|move|'+opp+'|Encore|'+own, '|-start|'+own+'|Encore','|upkeep','|turn|21'),
        fail+('|move|'+opp+'|Wrap|'+own+'|[miss]', '|-miss|'+opp+'|'+own,'|upkeep','|turn|22'),
    ]
    factory.state.history += sum(chunks, ())
    return factory


def plan_fixture():
    factory = ten_turn_factory()
    with mock.patch(MODULE+'.public_history',side_effect=lambda s:s.history), \
         mock.patch('pokezero.mcts_eval.paper_reference_staged_chance.public_history',side_effect=lambda s:s.history), \
         mock.patch('pokezero.mcts_eval.paper_reference_staged_chance.verify_compiled_tree'), \
         mock.patch('pokezero.mcts_eval.paper_reference_staged_chance.ENGINE_HASHES',{}):
        return build_wake_wrap_plan(factory)


def tail_snapshot(stage):
    snapshot = support_snapshot(0)
    own, opp = [side['pokemon'][0] for side in snapshot.bridge_snapshot['battle']['sides']]
    own['volatiles'] = dict(substitute=dict(hp=54 if stage < 8 else 42))
    own['hp'] = {5:246,6:262}.get(stage,264)
    own['lastMove'] = '[DataMove:psychic]' if stage == 5 else dict(move='[Move:substitute]',hit=0,totalDamage=False)
    if stage == 9:
        own['volatiles']['encore'] = dict(move='substitute',duration=3)
    if stage < 8:
        opp.update(status='slp',statusState=dict(time=8-stage,startTime=3,skippedTime=0,source='[Pokemon:p2a]'))
    return snapshot


class WakeWrapPlanTests(unittest.TestCase):
    def build(self, factory):
        with mock.patch(MODULE+'.public_history',side_effect=lambda s:s.history), \
             mock.patch('pokezero.mcts_eval.paper_reference_staged_chance.public_history',side_effect=lambda s:s.history), \
             mock.patch('pokezero.mcts_eval.paper_reference_staged_chance.verify_compiled_tree'), \
             mock.patch('pokezero.mcts_eval.paper_reference_staged_chance.ENGINE_HASHES',{}):
            return build_wake_wrap_plan(factory)

    def test_structural_ten_turn_certificate_and_history_not_seed_whitelist(self):
        plan = self.build(ten_turn_factory())
        self.assertIsInstance(plan, WakeWrapChancePlan)
        self.assertEqual(plan.receipt['schema'],WAKE_WRAP_SCHEMA)
        self.assertEqual(plan.own_actions,(3,)*5+(0,)*5)
        self.assertEqual(plan.receipt['public_stages'],10)
        self.assertNotIn('seed',plan.__dict__)
        self.assertEqual(plan.prefix.receipt['public_stages'],5)

    def test_later_suffix_is_still_original_joint_and_new_schema_not_relabelled(self):
        factory = ten_turn_factory()
        factory.pending_transition.continuation_actions += (None,)
        factory.state.history += ('|move|p2a: Shuckle|Toxic|p1a: Lugia','|upkeep','|turn|23')
        plan = self.build(factory)
        self.assertIsInstance(plan,StagedPrefixJointPlan)
        self.assertEqual(plan.receipt['schema'],WAKE_WRAP_JOINT_SCHEMA)
        self.assertEqual(plan.receipt['certified_prefix_stages'],10)
        self.assertEqual(plan.receipt['public_stages'],11)
        self.assertFalse(plan.receipt['suffix_chance_conditioning'])

    def test_unproven_public_variants_are_not_locally_conditioned(self):
        changes = [
            ('|-activate|p1a: Lugia|Substitute|[damage]','|-end|p1a: Lugia|Substitute'),
            ('|-start|p1a: Lugia|Encore','|-fail|p2a: Shuckle'),
            ('|move|p2a: Shuckle|Wrap|p1a: Lugia|[miss]','|move|p2a: Shuckle|Wrap|p1a: Lugia'),
            ('|cant|p2a: Shuckle|slp','|-curestatus|p2a: Shuckle|slp'),
            ('|-heal|p1a: Lugia|262/264|[from] item: Leftovers','|-heal|p1a: Lugia|262/264|[from] item: Black Sludge'),
        ]
        for old, new in changes:
            factory = ten_turn_factory()
            factory.state.history = tuple(new if line == old else line for line in factory.state.history)
            self.assertIsNone(self.build(factory),new)
        factory = ten_turn_factory()
        factory.pending_transition.continuation_actions = (3,3,3,3,0,0,0,1,0)
        self.assertIsNone(self.build(factory))

    def test_seat_names_and_slot_indices_are_not_fixed(self):
        factory = ten_turn_factory()
        factory.state.player_id = 'p2'
        factory.state.history = tuple(line.replace('p1a: Lugia','p2a: Lugia').replace('p2a: Shuckle','p1a: Shuckle')
            for line in factory.state.history)
        factory.pending_transition.own_action = 1
        factory.pending_transition.continuation_actions = (1,)*4+(2,)*5
        plan = self.build(factory)
        self.assertEqual((plan.subject,plan.opponent),('p2','p1'))
        self.assertEqual(plan.own_actions,(1,)*5+(2,)*5)


class WakeWrapSupportTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan_fixture()

    def test_original_five_stage_support_and_zero_likelihood_timer_are_retained(self):
        for stage in range(5):
            self.assertTrue(validate_active_support(self.plan,support_snapshot(stage),stage))
        snapshot = support_snapshot(0)
        snapshot.bridge_snapshot['battle']['sides'][0]['pokemon'][0]['volatiles']['encore']['duration']=6
        self.assertFalse(validate_active_support(self.plan,snapshot,0))

    def test_all_tail_stages_and_minimum_survival_bound(self):
        for stage in range(5,10):
            self.assertTrue(validate_active_support(self.plan,tail_snapshot(stage),stage))
        for duration in range(3,7):
            snapshot = tail_snapshot(9)
            snapshot.bridge_snapshot['battle']['sides'][0]['pokemon'][0]['volatiles']['encore']['duration']=duration
            self.assertTrue(validate_active_support(self.plan,snapshot,9))

    def test_active_move_serialization_is_bound_not_treated_as_static_data_move(self):
        for last_move in ('[DataMove:substitute]',dict(move='[Move:substitute]',hit=0,totalDamage=False)):
            snapshot=tail_snapshot(8)
            snapshot.bridge_snapshot['battle']['sides'][0]['pokemon'][0]['lastMove']=last_move
            self.assertTrue(validate_active_support(self.plan,snapshot,8))
        for last_move in (dict(move='[Move:recover]',hit=0,totalDamage=False),
                          dict(move='[Move:substitute]',hit=0,totalDamage=9),
                          dict(move='[Move:substitute]',hit=0,totalDamage=False,flags=dict(failencore=True))):
            snapshot=tail_snapshot(8)
            snapshot.bridge_snapshot['battle']['sides'][0]['pokemon'][0]['lastMove']=last_move
            with self.assertRaisesRegex(ReferenceRefusal,'last-move'):
                validate_active_support(self.plan,snapshot,8)

    def test_hidden_bench_order_positive_opponent_PP_and_policy_mass_not_restricted(self):
        for stage in range(5,10):
            snapshot=tail_snapshot(stage)
            for side in snapshot.bridge_snapshot['battle']['sides']:
                side['pokemon'].extend([dict(species='unknown bench two'),dict(species='unknown bench one')])
            opp=snapshot.bridge_snapshot['battle']['sides'][1]['pokemon'][0]
            opp['moveSlots'].reverse()
            for slot in opp['moveSlots']:
                slot['pp']=1
            self.assertTrue(validate_active_support(self.plan,snapshot,stage))
        for stage in (5,6):
            for move in ('wrap','rest','toxic','encore'):
                self.assertTrue(self.plan.compatible_move(stage,dict(kind='move',legal=True,move_id=move)))
            self.assertFalse(self.plan.compatible_move(stage,dict(kind='switch',legal=True)))
        for stage,move in ((7,'wrap'),(8,'encore'),(9,'wrap')):
            self.assertTrue(self.plan.compatible_move(stage,dict(kind='move',legal=True,move_id=move)))
            self.assertFalse(self.plan.compatible_move(stage,dict(kind='move',legal=True,move_id='rest')))

    def test_unsupported_latents_refuse_not_redraw(self):
        edits = [
            (7,lambda own,opp,b:own['volatiles']['substitute'].update(hp=12)),
            (8,lambda own,opp,b:own['volatiles']['substitute'].update(hp=41)),
            (8,lambda own,opp,b:own.update(hp=263)),
            (7,lambda own,opp,b:opp.update(hp=197)),
            (6,lambda own,opp,b:opp['statusState'].update(time=1)),
            (6,lambda own,opp,b:opp['statusState'].update(source='[Pokemon:p1a]')),
            (7,lambda own,opp,b:own.update(lastMove='[DataMove:recover]')),
            (8,lambda own,opp,b:own['moveSlots'][0].update(pp=2)),
            (9,lambda own,opp,b:own['volatiles']['encore'].update(duration=2)),
            (9,lambda own,opp,b:own['volatiles']['encore'].update(move='psychic')),
            (8,lambda own,opp,b:opp.update(ability='earlybird')),
            (8,lambda own,opp,b:b['field'].update(weather='sandstorm')),
            (8,lambda own,opp,b:own['volatiles'].update(protect={})),
        ]
        for stage, edit in edits:
            snapshot=tail_snapshot(stage)
            battle=snapshot.bridge_snapshot['battle']
            own,opp=[side['pokemon'][0] for side in battle['sides']]
            edit(own,opp,battle)
            with self.assertRaises(ReferenceRefusal,msg=str((stage,edit))):
                validate_active_support(self.plan,snapshot,stage)

    def test_exact_finite_posterior_keeps_all_ten_hidden_dependent_policy_factors(self):
        joint, staged, forced_policy = {}, {}, {}
        chance=[Fraction(1,8),Fraction(1,16),Fraction(1,256),Fraction(1,256),Fraction(1,16),
            Fraction(1),Fraction(1),Fraction(85,100),Fraction(1),Fraction(15,100)]
        common,retry=Fraction(1),Fraction(1)
        for probability in chance:
            common *= probability
            retry *= 1-(1-probability)**7
        for theta in range(3):
            for private in range(3):
                policy=Fraction(theta+1,4)*Fraction(private+1,5)
                # Wrap / Encore / Wrap policy mass depends on the latent team.
                tail=Fraction(theta+2,5)*Fraction(private+2,6)*Fraction(theta+private+1,7)
                joint[theta,private]=policy*tail*common
                staged[theta,private]=policy*tail*retry
                forced_policy[theta,private]=policy*retry
        normalize=lambda rows:{key:value/sum(rows.values()) for key,value in rows.items()}
        self.assertEqual(normalize(joint),normalize(staged))
        self.assertNotEqual(normalize(joint),normalize(forced_policy))


if __name__ == '__main__':
    unittest.main()
