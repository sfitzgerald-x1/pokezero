from fractions import Fraction
import random
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.paper_reference import ReferenceRefusal
from pokezero.mcts_eval.paper_reference_anchor_constraints import prepare_constraints, anchor_matches, conditional_encore_support
from pokezero.mcts_eval.paper_reference_seed_bank import encore_hints, matches_hints, native_seed_outputs
from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory


class AnchorConstraintsTests(unittest.TestCase):
    def prepare(self, history, suffix, certificate=None, pp=10):
        state=NS(history=tuple(history),self_request={'active':[{'moves':[{'id':'psychic','pp':pp}]}]})
        with patch('pokezero.mcts_eval.paper_reference_anchor_constraints.public_history',lambda s:s.history), \
             patch('pokezero.local_showdown._public_reference_encore',return_value=certificate):
            return prepare_constraints(state,tuple(history+suffix),'p1')

    def test_public_gender_level_are_necessary_without_forcing_other_traits(self):
        c=self.prepare(['|turn|1'],['|switch|p2a: Marowak|Marowak, L83, M|100/100','|turn|2'])
        self.assertEqual(c['static_details'],[dict(species='marowak',gender='M',level=83)])
        env=NS(_search_snapshot_permitted=True,snapshot=lambda:NS(bridge_snapshot={'battle':{'sides':[
            {'id':'p2','pokemon':[{'details':'Marowak, L83, M'}]}]}}))
        self.assertIsNone(anchor_matches(env,{},c))
        env.snapshot=lambda:NS(bridge_snapshot={'battle':{'sides':[
            {'id':'p2','pokemon':[{'details':'Marowak, L83, F'}]}]}})
        self.assertIn('static details',anchor_matches(env,{},c))

    def test_species_changes_and_conflicting_reveals_disable_hard_details(self):
        switch='|switch|p2a: Marowak|Marowak, L83, M|100/100'
        for extra in ('|move|p2a: Marowak|Transform|p1a: Lugia',
                      '|-formechange|p2a: Marowak|Ditto',
                      '|switch|p2a: Marowak|Marowak, L83, F|100/100'):
            self.assertEqual(self.prepare([], [switch,extra])['static_details'],[])

    def test_live_hidden_world_is_rejected_before_snapshot(self):
        c=self.prepare([],[])
        with self.assertRaisesRegex(ReferenceRefusal,'owned hypothetical'):
            anchor_matches(NS(_search_snapshot_permitted=False),{},c)

    def test_own_encore_clock_guarded_by_known_pp_and_no_disruptions(self):
        h=['|switch|p1a: Lugia|Lugia, L70|100/100','|turn|3']
        s=['|move|p1a: Lugia|Psychic|p2a: Shuckle','|upkeep',
           '|turn|4','|move|p1a: Lugia|Psychic|p2a: Shuckle',
           '|-end|p1a: Lugia|Encore','|upkeep','|turn|5']
        certificate=dict(move='psychic',remaining_candidates=[1,2,3])
        c=self.prepare(h,s,certificate)
        self.assertEqual(c['own_encore']['required_remaining'],2)
        self.assertEqual(c['own_encore']['maximum_pp_payments'],4)
        self.assertIsNone(self.prepare(h,s,certificate,pp=4)['own_encore'])
        for event in ('|move|p2a: Shuckle|Spite|p1a: Lugia','|faint|p2a: Shuckle',
                      '|switch|p1a: Snorlax|Snorlax, L80, M|100/100'):
            self.assertIsNone(self.prepare(h,[event,*s],certificate)['own_encore'])
        env=NS(_search_snapshot_permitted=True)
        self.assertIsNone(anchor_matches(env,{'encore_conditioning':{'p1':{'sampled_remaining':2}}},c))
        self.assertIn('Encore',anchor_matches(env,{'encore_conditioning':{'p1':{'sampled_remaining':1}}},c))

    def test_conditional_necessary_predicate_normalizer_cancels_exactly(self):
        prior={(g,t):Fraction(1,8) for g in ('M','F') for t in range(1,5)}
        likelihood={(g,t):Fraction(t,10) if g=='M' and t==2 else Fraction(0) for g,t in prior}
        def normalized(m):
            total=sum(m.values()); return {k:v/total for k,v in m.items()}
        original=normalized({k:p*likelihood[k] for k,p in prior.items()})
        conditional={k:p for k,p in prior.items() if k[0]=='M' and k[1]==2}
        z=sum(conditional.values())
        early=normalized({k:conditional.get(k,0)/z*likelihood[k] for k in prior})
        self.assertEqual(original,early)

    def test_direct_uniform_timer_kernel_matches_rejection_conditioning(self):
        for original in ([1],[1,2,3],[3,4,5,6]):
            for required in original:
                support,evidence=conditional_encore_support(original,required)
                self.assertEqual(support,[required])
                self.assertEqual(evidence['removed_constant_probability'],1/len(original))
                for prior_mass in (Fraction(1,8),Fraction(3,8)):
                    original_joint=prior_mass*Fraction(1,len(original))
                    self.assertEqual(original_joint/Fraction(1,len(original)),prior_mass)
        for original,required in (([],1),([1,1],1),([1,2],3),([True],True)):
            with self.assertRaisesRegex(ReferenceRefusal,'original timer kernel'):
                conditional_encore_support(original,required)

    def test_default_off_and_requires_particles(self):
        self.assertFalse(ShowdownWorkerFactory('a','b','c','d').public_anchor_constraints)
        for value in (True,1,'true'):
            with self.assertRaisesRegex(ReferenceRefusal,'public anchor constraints'):
                ShowdownWorkerFactory('a','b','c','d',public_anchor_constraints=value)


class EncoreHintTests(unittest.TestCase):
    def test_native_context_duration_hint_and_seed_projection(self):
        h=('|turn|3',)
        future=('|-start|p1a: Lugia|Encore','|upkeep','|turn|4','|upkeep',
                '|turn|5','|upkeep','|turn|6','|-end|p1a: Lugia|Encore')
        for offset in (False,True):
            trace=[dict(ordinal=1,range=[3,7],chance=None,
                volatile=dict(id='encore',ident='p1a: Lugia',after_target_acted=offset))]
            hints=encore_hints(trace,h,h+future)
            self.assertEqual(hints[0]['support'],[4-int(offset)])
            for seed in range(20):
                roll=3+native_seed_outputs(seed,1)[0]*4//(1<<32)
                self.assertEqual(matches_hints(seed,hints),roll==4-int(offset))

    def test_later_episode_or_switch_is_not_hint_for_current_pilot(self):
        h=('|turn|3',)
        trace=[dict(ordinal=1,range=[3,7],chance=None,
            volatile=dict(id='encore',ident='p1a: Lugia',after_target_acted=False))]
        for future in (('|turn|4','|-start|p1a: Lugia|Encore'),
            ('|-start|p1a: Lugia|Encore','|switch|p1a: Lugia|Lugia, L70|100/100','|-end|p1a: Lugia|Encore')):
            self.assertEqual(encore_hints(trace,h,h+future),[])

    def test_missing_or_non_encore_native_context_is_not_invented(self):
        h=('|turn|3',)
        future=h+('|-start|p1a: Lugia|Encore','|-end|p1a: Lugia|Encore')
        for row in ({},dict(range=[3,7],volatile={'id':'confusion'})):
            self.assertEqual(encore_hints([row],h,future),[])


if __name__=='__main__': unittest.main()
