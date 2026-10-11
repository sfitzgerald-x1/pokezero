"""Probability, censorship and ownership tests; no live game or model required."""
from dataclasses import dataclass
import math
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pokezero.mcts_eval.paper_reference import ReferenceRefusal, SamplingDeadlineExceeded
from pokezero.mcts_eval.paper_reference_particles import (
    bootstrap_population, release_unique, validate_particle_count, HypotheticalHistoryPopulation)
from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory


@dataclass(frozen=True)
class Toy:
    hypothesis: str
    generation: int


class ParticleTests(unittest.TestCase):
    def test_indicator_filter_targets_likelihood_not_uniform_feasibility_or_prior(self):
        # q(A)=.8, q(B)=.2; L(A)=.2*.1, L(B)=.8*.9.
        # Posterior B=.144/(.016+.144)=.9, not prior .2 or feasible-uniform .5.
        estimates = []
        for seed in range(3):
            rng, receipt = random.Random(seed), {}
            def initial(_):
                return Toy('A' if rng.random() < .8 else 'B', 0)
            def advance(p, stage):
                probability = ((.2, .1) if p.hypothesis == 'A' else (.8, .9))[stage]
                return Toy(p.hypothesis, stage+1) if rng.random() < probability else None
            particles = bootstrap_population(count=4096, stages=2, initial=initial,
                advance=advance, release=lambda p: None, rng=rng, check=lambda: None, receipt=receipt)
            estimates.append(sum(p.hypothesis == 'B' for p in particles)/len(particles))
            self.assertEqual(receipt['initialized'], 4096)
            self.assertTrue(receipt['complete'])
            self.assertFalse(receipt['exact_posterior'])
            self.assertTrue(receipt['stages'][0]['resampled'])
            self.assertFalse(receipt['stages'][1]['resampled'])
        self.assertAlmostEqual(sum(estimates)/3, .9, delta=.025)

    def test_prefix_work_is_retained_without_reinitializing_anchors(self):
        initialized, counts = [], [0, 0, 0]
        def initial(i):
            initialized.append(i)
            return Toy(str(i), 0)
        def advance(p, stage):
            counts[stage] += 1
            self.assertEqual(p.generation, stage)
            return Toy(p.hypothesis, stage+1)
        particles = bootstrap_population(count=4, stages=3, initial=initial, advance=advance,
            release=lambda p: None, rng=random.Random(1), check=lambda: None, receipt={})
        self.assertEqual(initialized, list(range(4)))
        self.assertEqual(counts, [4,4,4])
        self.assertEqual(len(particles), 4)

    def test_no_survivors_refuses_without_retry_or_feasibility_fallback(self):
        created, freed, receipt = [], [], {}
        def initial(i):
            p = Toy(str(i), 0); created.append(p); return p
        with self.assertRaisesRegex(ReferenceRefusal, 'zero matching survivors') as failure:
            bootstrap_population(count=4, stages=2, initial=initial, advance=lambda p,s: None,
                release=freed.append, rng=random.Random(1), check=lambda: None, receipt=receipt)
        self.assertEqual(created, freed)
        self.assertFalse(receipt['complete'])
        self.assertTrue(receipt['stages'][0]['stage_complete'])
        self.assertEqual(failure.exception.sampling_diagnostic['bootstrap_population']['stages'][0]['survivors'], 0)

    def test_deadline_partial_stage_releases_inputs_and_matching_outputs(self):
        created, freed, receipt = [], [], {}
        calls = 0
        def check():
            nonlocal calls
            calls += 1
            if calls == 12:  # initialization8; then two partial outputs
                raise SamplingDeadlineExceeded('test deadline')
        def initial(i):
            p = Toy(str(i), 0);created.append(p);return p
        def advance(p, stage):
            p = Toy(p.hypothesis, 1);created.append(p);return p
        with self.assertRaises(SamplingDeadlineExceeded):
            bootstrap_population(count=4, stages=2, initial=initial, advance=advance,
                release=freed.append, rng=random.Random(1), check=check, receipt=receipt)
        self.assertCountEqual([id(p) for p in freed], [id(p) for p in created])
        self.assertEqual(len(freed), len(created))
        self.assertFalse(receipt['stages'][0]['stage_complete'])
        self.assertFalse(receipt['partial_stage_used'])
        self.assertFalse(receipt['complete'])

    def test_native_error_is_not_reinterpreted_as_zero_observation_weight(self):
        freed = []
        def broken(p, s):
            raise RuntimeError('bridge broke')
        with self.assertRaisesRegex(RuntimeError, 'bridge broke'):
            bootstrap_population(count=3, stages=1, initial=lambda i: Toy(str(i),0),
                advance=broken, release=freed.append, rng=random.Random(1), check=lambda:None, receipt={})
        self.assertEqual(len(freed),3)

    def test_resampled_owners_release_once_even_with_aliases(self):
        p = Toy('A', 0)
        calls = []
        release_unique([p,p,p], calls.append)
        self.assertEqual(calls, [p])

    def test_runtime_opt_in_is_integer_bounded_and_default_off(self):
        self.assertEqual(ShowdownWorkerFactory('c','d','e','s').history_particles, 0)
        for bad in (True, False, -1, 1, 15, 257, 32.0, '32'):
            with self.subTest(value=bad), self.assertRaises(ReferenceRefusal):
                validate_particle_count(bad)
        for good in (0,16,32,256):
            self.assertEqual(validate_particle_count(good),good)

    def test_population_cache_owns_only_hypothetical_snapshots_and_reuses_matching_prefixes(self):
        class Env:
            _search_snapshot_permitted = True
            def __init__(self):
                self.stage=0; self.next_id=0; self.handles={}; self.transitions=0
            def snapshot_for_search(self):
                self.next_id+=1;self.handles[self.next_id]=self.stage;return self.next_id
            def release_search_snapshot(self, handle):
                return self.handles.pop(handle, None) is not None
            def restore_search_snapshot(self, handle):
                self.stage=self.handles[handle]
            def requested_players(self):
                return ('p1','p2')
            def observe(self, player):
                return SimpleNamespace(legal_action_mask=(True,True), stage=self.stage,
                    metadata={'action_candidates':[
                        dict(action_index=0,kind='move',move_id='toxic'),
                        dict(action_index=1,kind='move',move_id='splash')]})
            def public_materialization_state(self, player):
                return SimpleNamespace(history=tuple(range(self.stage+1)), deferred_opponent_action_player=None)
            def step_from_search_snapshot_for_conditioning(self, snapshot, actions, chance_seed):
                self.stage=self.handles[snapshot]+1;self.transitions+=1
            def terminal(self):
                return None
            def snapshot(self):
                return SimpleNamespace(bridge_snapshot={'battle':{'sides':[
                    {'id':'p1','pokemon':[{'volatiles':{'substitute':{'hp':17}}}]},
                    {'id':'p2','pokemon':[{'volatiles':{}}]}]}})
        env=Env()
        class Prior:
            def __init__(self, **kwargs):
                self.receipts=[]
            def bind_sampling_deadline(self, deadline):
                pass
            def __call__(self, rng):
                env.stage=0;self.receipts.append({'status':'ROOT_VALIDATED'})
                return SimpleNamespace(close=lambda:None)
        state=SimpleNamespace(player_id='p1', history=(0,1,2), deferred_opponent_action_player=None)
        transition=SimpleNamespace(before_state=None,before_observation=None, own_action=0,
            continuation_actions=(1,), prior_transition=None)
        factory=SimpleNamespace(env=env,state=state,pending_transition=transition,
            evaluator=lambda observation:((0,1),SimpleNamespace(priors=(.9,.1))),
            set_source=None, allow_earlier_compatible_template=True,max_known_set_draws=128,
            sampling_deadline_at=None,history_particles=16,root=SimpleNamespace(key=b'root'),
            check_sampling_deadline=lambda:None,_release=lambda evidence:evidence.update(released=True))
        population=HypotheticalHistoryPopulation(factory)
        with patch('pokezero.mcts_eval.paper_reference_factory.PublicRootWorldFactory',Prior), \
             patch('pokezero.mcts_eval.paper_reference_particle_party.condition_necessary_party',return_value={}), \
             patch('pokezero.mcts_eval.paper_reference_particles.public_history',side_effect=lambda s:s.history), \
             patch('pokezero.mcts_eval.paper_reference_particles.decision_state',return_value=factory.root):
            receipts=[]
            for seed in (1,2,3):
                evidence={};population.draw(random.Random(seed),evidence).close();receipts.append(evidence)
            self.assertEqual(env.transitions,32)  # not96: build once, retain prefixes
            self.assertEqual(len(env.handles),16)
            self.assertEqual([r['substitute_particle_conditioning']['empirical_draw'] for r in receipts],[1,2,3])
            self.assertTrue(all(r['released'] for r in receipts))
            self.assertTrue(all(not r['substitute_particle_conditioning']['exact_posterior'] for r in receipts))
            self.assertTrue(all('phase_timing' not in r['substitute_particle_conditioning'] for r in receipts))
            population.close();self.assertEqual(env.handles,{})

        factory.history_chance_pool=4
        population=HypotheticalHistoryPopulation(factory)
        before=env.transitions
        with patch('pokezero.mcts_eval.paper_reference_factory.PublicRootWorldFactory',Prior), \
             patch('pokezero.mcts_eval.paper_reference_particle_party.condition_necessary_party',return_value={}), \
             patch('pokezero.mcts_eval.paper_reference_particles.public_history',side_effect=lambda s:s.history), \
             patch('pokezero.mcts_eval.paper_reference_particles.decision_state',return_value=factory.root):
            first={};population.draw(random.Random(1),first).close()
            second={};population.draw(random.Random(2),second).close()
            self.assertEqual(env.transitions-before,128)  # 16 particles x2 stages x4 original chance draws
            pools=first['substitute_particle_conditioning']['chance_pools']
            self.assertEqual(len(pools),32)
            self.assertTrue(all(p['completed']==4 and p['matches']==4 and p['pool_complete'] for p in pools))
            self.assertNotIn('chance_pools',second['substitute_particle_conditioning'])
            self.assertEqual(second['substitute_particle_conditioning']['chance_pools_retained_at_empirical_draw'],1)
            for particle in population.particles:
                self.assertTrue(all(step['chance_pool']['observation_likelihood_estimate']==1. for step in particle.steps))
            population.close();self.assertEqual(env.handles,{})
        factory.history_chance_pool=1
        # The weighted adapter path must preserve pi/q all the way into the
        # cached empirical draw, not just pass the standalone oracle.
        factory.guide_history_actions=True
        population=HypotheticalHistoryPopulation(factory)
        with patch('pokezero.mcts_eval.paper_reference_factory.PublicRootWorldFactory',Prior), \
             patch('pokezero.mcts_eval.paper_reference_particle_party.condition_necessary_party',return_value={}), \
             patch('pokezero.mcts_eval.paper_reference_particles.public_history',side_effect=lambda s:s.history), \
             patch('pokezero.mcts_eval.paper_reference_particles.decision_state',return_value=factory.root), \
             patch('pokezero.mcts_eval.paper_reference_action_proposal.public_guidance_actions',return_value=((0,),{})), \
             patch('pokezero.mcts_eval.paper_reference_stage_inputs.public_history',side_effect=lambda s:s.history):
            evidence={}
            population.draw(random.Random(5),evidence).close()
            receipt=evidence['substitute_particle_conditioning']
            self.assertTrue(receipt['action_importance_correction'])
            self.assertAlmostEqual(sum(receipt['final_normalized_weights']),1.)
            self.assertGreaterEqual(receipt['final_particle_ess'],1.)
            self.assertLessEqual(receipt['final_ancestry_ess'],receipt['distinct_anchor_ancestors']+1e-12)
            for particle in population.particles:
                for step in particle.steps:
                    proposal=step['action_proposal']
                    self.assertTrue(proposal['full_policy_support'])
                    self.assertAlmostEqual(proposal['importance_weight'],
                        proposal['selected_original_probability']/proposal['selected_proposal_probability'])
            # The stored bank does not copy live game/opponent choices.
            self.assertFalse(receipt['live_hidden_state_used'])
            self.assertFalse(receipt['live_opponent_action_used'])
            population.close();self.assertEqual(env.handles,{})

        # Exercise the production memo gate, not only the generic input helper.
        # Canonical deterministic policy calls and projections may be reused,
        # while every multiplicity still gets its own native chance transition.
        from pokezero.mcts_eval.paper_reference_showdown import ChampionEvaluator
        factory.evaluator=object.__new__(ChampionEvaluator)
        factory.guide_history_actions=False
        factory.collect_phase_timing=True
        before=env.transitions
        population=HypotheticalHistoryPopulation(factory)
        with patch('pokezero.mcts_eval.paper_reference_factory.PublicRootWorldFactory',Prior), \
             patch('pokezero.mcts_eval.paper_reference_particle_party.condition_necessary_party',return_value={}), \
             patch('pokezero.mcts_eval.paper_reference_particles.public_history',side_effect=lambda s:s.history), \
             patch('pokezero.mcts_eval.paper_reference_particles.decision_state',return_value=factory.root), \
             patch.object(ChampionEvaluator,'__call__',return_value=((0,1),SimpleNamespace(priors=(.9,.1)))) as evaluate:
            evidence={}
            population.draw(random.Random(5),evidence).close()
            receipt=evidence['substitute_particle_conditioning']
            memo=receipt['historical_input_memo']
            self.assertTrue(memo['enabled'])
            self.assertGreater(memo['hits'],0)
            self.assertEqual(evaluate.call_count,memo['misses'])
            self.assertEqual(env.transitions-before,32)
            self.assertEqual(len(receipt['chance_pools']),32)
            self.assertEqual(len(receipt['initial_anchor_proposals']),16)
            timing=receipt['phase_timing']
            for name,calls in (('initial_anchor_inclusive',16),
                    ('stage_inputs_inclusive',32),('native_chance_transition',32),
                    ('post_transition_public_capture',32),('survivor_snapshot',32)):
                self.assertEqual(timing[name]['calls'],calls)
                self.assertGreaterEqual(timing[name]['seconds'],0.)
                self.assertTrue(math.isfinite(timing[name]['seconds']))
            population.close();self.assertEqual(env.handles,{})


        # Failed pools must retain the same pre-trial action proposal as accepted
        # paths, not just an ambiguous first public mismatch. No extra draws.
        factory.evaluator=lambda observation:((0,1),SimpleNamespace(priors=(.9,.1)))
        factory.guide_history_actions=True
        population=HypotheticalHistoryPopulation(factory)
        original_step=env.step_from_search_snapshot_for_conditioning
        def mismatching_step(*args, **kwargs):
            original_step(*args, **kwargs)
            env.stage=99
        with patch('pokezero.mcts_eval.paper_reference_factory.PublicRootWorldFactory',Prior), \
             patch('pokezero.mcts_eval.paper_reference_particle_party.condition_necessary_party',return_value={}), \
             patch('pokezero.mcts_eval.paper_reference_particles.public_history',side_effect=lambda s:s.history), \
             patch('pokezero.mcts_eval.paper_reference_particles.decision_state',return_value=factory.root), \
             patch('pokezero.mcts_eval.paper_reference_action_proposal.public_guidance_actions',return_value=((0,),{'identifier':'toy'})), \
             patch('pokezero.mcts_eval.paper_reference_stage_inputs.public_history',side_effect=lambda s:s.history), \
             patch.object(env,'step_from_search_snapshot_for_conditioning',side_effect=mismatching_step):
            with self.assertRaisesRegex(ReferenceRefusal,'zero matching survivors') as failure:
                population.draw(random.Random(5),{})
            pools=failure.exception.sampling_diagnostic['bootstrap_population']['chance_pools']
            self.assertEqual(len(pools),16)
            for pool in pools:
                pre=pool['pre_trial_proposal']
                self.assertTrue(pre['history_available'])
                self.assertEqual(pre['history_length'],1)
                self.assertEqual(pre['history_tail'],[0])
                self.assertEqual(pre['expected_suffix'],[1,2])
                self.assertEqual(pre['opponent_legal'],[0,1])
                self.assertEqual(pre['opponent_priors'],[.9,.1])
                self.assertEqual([r['move_id'] for r in pre['opponent_action_candidates']],['toxic','splash'])
                proposal=pre['action_proposal']
                self.assertEqual(proposal['guidance'],{'identifier':'toy'})
                self.assertEqual(proposal['compatible_actions'],[0])
                self.assertAlmostEqual(proposal['importance_weight'],
                    proposal['selected_original_probability']/proposal['selected_proposal_probability'])
                self.assertIn(pre['opponent_action'],pre['opponent_legal'])
                self.assertEqual(pool['rejection_reasons'],{'different exact public prefix':1})
            population.close();self.assertEqual(env.handles,{})


if __name__ == '__main__':
    unittest.main()
