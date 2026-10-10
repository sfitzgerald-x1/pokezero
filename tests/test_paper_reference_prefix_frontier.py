"""Sequential posterior weighting, prefix identity and handle ownership."""
from dataclasses import replace
import random
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.paper_reference import ReferenceRefusal, SamplingDeadlineExceeded
from pokezero.mcts_eval.paper_reference_particles import bootstrap_population, HypotheticalHistoryPopulation, HistoryParticle
from pokezero.mcts_eval.paper_reference_prefix_frontier import compatible_complete_prefix, OPTIONS
from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
from pokezero.mcts_eval.paper_reference_substitute import SubstituteHistoryTransition


class FrontierTests(unittest.TestCase):
    def test_previous_weights_and_new_likelihood_are_both_preserved(self):
        estimates = []
        for seed in range(4):
            receipt, calls = {}, []
            def advance(p, stage):
                self.assertEqual(stage, 2)
                return p + '-child' if rng.random() < (.9 if p == 'A' else .1) else None
            rng = random.Random(seed)
            particles = bootstrap_population(count=4096, stages=1,
                initial=lambda _: self.fail('old anchors redrawn'), advance=advance,
                release=calls.append, rng=rng, check=lambda: None, receipt=receipt,
                frontier=['A', 'B'], frontier_weights=[.1, .9], stage_offset=2)
            estimates.append(sum(p.startswith('B') for p in particles)/len(particles))
            self.assertEqual(receipt['fresh_initialization_calls'], 0)
            self.assertEqual(receipt['stages'][0]['stage'], 2)
            self.assertTrue(receipt['complete'])
        # .1*.9 == .9*.1. Ignoring prior weights or applying them twice is wrong.
        self.assertAlmostEqual(sum(estimates)/len(estimates), .5, delta=.035)

    def test_zero_survivors_refuses_and_frees_retained_owners_once(self):
        owners = [object(), object()]
        freed, receipt = [], {}
        with self.assertRaisesRegex(ReferenceRefusal, 'zero matching survivors'):
            bootstrap_population(count=32, stages=1, initial=None,
                advance=lambda p, s: None, release=freed.append, rng=random.Random(5),
                check=lambda: None, receipt=receipt, frontier=owners,
                frontier_weights=[.4, .6], stage_offset=2)
        self.assertCountEqual(freed, owners)
        self.assertFalse(receipt['complete'])
        self.assertFalse(receipt['partial_stage_used'])

    def test_partial_update_never_exposes_old_or_new_particles(self):
        owners = [object(), object()]
        freed, receipt, calls = [], {}, 0
        def check():
            nonlocal calls
            calls += 1
            if calls == 5:
                raise SamplingDeadlineExceeded('partial incremental update')
        def advance(p, s):
            child = object()
            owners.append(child)
            return child
        with self.assertRaises(SamplingDeadlineExceeded):
            bootstrap_population(count=32, stages=2, initial=None, advance=advance,
                release=freed.append, rng=random.Random(5), check=check, receipt=receipt,
                frontier=owners[:], frontier_weights=[.4, .6], stage_offset=2)
        self.assertCountEqual(freed, owners)
        self.assertFalse(receipt['stages'][0]['stage_complete'])
        self.assertFalse(receipt['partial_stage_used'])

    def test_invalid_retained_weights_fail_closed_and_release_owners(self):
        for weights in ([.1, .1], [0., 1.], [float('nan'), .5], [1.]):
            owners, freed = [object(), object()], []
            with self.assertRaisesRegex(ReferenceRefusal, 'invalid final weights'):
                bootstrap_population(count=32, stages=1, initial=None,
                    advance=lambda p, s: p, release=freed.append, rng=random.Random(2),
                    check=lambda: None, receipt={}, frontier=owners,
                    frontier_weights=weights, stage_offset=2)
            self.assertCountEqual(freed, owners)

    def setup_pair(self):
        history = lambda lines: S(requests={}, public_events=[S(raw_line=line) for line in lines])
        anchor = S(player_id='p1', format_id='gen3randombattle',
            observation_format_id='gen3randombattle', replay=history(['|turn|1']),
            self_request={'side': {'id': 'p1'}}, self_initial_request={}, self_move_states={},
            self_recharge_pp_charge={}, belief_engine=S(snapshot=lambda: {'public': 'anchor'}))
        env, evaluator, root = S(release_search_snapshot=lambda _: True), object(), S(key=b'root')
        def factory(actions, lines):
            f = S(env=env, evaluator=evaluator, active=False, root=root,
                state=S(player_id='p1', replay=history(lines)),
                pending_transition=SubstituteHistoryTransition(anchor, S(key=b'anchor'),
                    actions[0], 'source', None, tuple(actions[1:])))
            for name in OPTIONS:
                setattr(f, name, 32 if name == 'history_particles' else False)
            return f
        old = HypotheticalHistoryPopulation(factory((1, 2), ['|turn|1', '|turn|2']))
        old.particles = [HistoryParticle(object(), i, {}, ({}, {})) for i in range(2)]
        old.receipt.update(complete=True, partial_stage_used=False, final_actor_root_key='726f6f74',
            final_normalized_weights=[.2, .8], stages=[{'stage': 0}, {'stage': 1}])
        new = factory((1, 2, 3), ['|turn|1', '|turn|2', '|turn|3'])
        return old, new

    def test_complete_prefix_transfers_ownership_and_retains_weight_certificate(self):
        old, new = self.setup_pair()
        particles = old.particles[:]
        with patch('pokezero.mcts_eval.paper_reference_prefix_frontier.decision_state',
                   side_effect=lambda obs, player: obs):
            self.assertTrue(compatible_complete_prefix(old, new))
            next_population = HypotheticalHistoryPopulation(new)
            self.assertTrue(next_population.adopt_complete_prefix(old))
        self.assertEqual(old.particles, [])
        self.assertEqual(next_population._retained_frontier, particles)
        self.assertEqual(next_population._retained_weights, [.2, .8])
        self.assertEqual(next_population._stage_offset, 2)
        self.assertFalse(next_population.receipt['complete'])
        self.assertFalse(next_population.receipt['retained_conditioning_prefix']['cold_stream_equivalence'])
        freed = []
        new.env.release_search_snapshot = lambda p: freed.append(p) or True
        old.close()
        next_population.close()
        self.assertEqual(freed, [p.snapshot for p in particles])

    def test_partial_drift_and_nonextension_banks_are_not_reused(self):
        for mutate in ('partial', 'kernel', 'actor', 'history', 'own_action', 'anchor', 'nested', 'active'):
            old, new = self.setup_pair()
            if mutate == 'partial': old.receipt['complete'] = False
            if mutate == 'kernel': new.history_particles = 64
            if mutate == 'actor': new.state.player_id = 'p2'
            if mutate == 'history': new.state.replay.public_events[1].raw_line = '|turn|99'
            if mutate == 'own_action': new.pending_transition = replace(new.pending_transition, own_action=99)
            if mutate == 'anchor':
                changed = S(**vars(new.pending_transition.before_state))
                changed.self_request = {'side': {'id': 'p2'}}
                new.pending_transition = replace(new.pending_transition, before_state=changed)
            if mutate == 'nested': new.pending_transition = replace(new.pending_transition, prior_transition=object())
            if mutate == 'active': old.factory.active = True
            particles = old.particles[:]
            with patch('pokezero.mcts_eval.paper_reference_prefix_frontier.decision_state',
                       side_effect=lambda obs, player: obs):
                self.assertFalse(HypotheticalHistoryPopulation(new).adopt_complete_prefix(old), mutate)
            self.assertEqual(old.particles, particles)

    def test_runtime_flag_requires_particles_and_defaults_off(self):
        self.assertFalse(ShowdownWorkerFactory('c', 'd', 'e', 's').history_prefix_reuse)
        for bad in (True, 1, 'true'):
            with self.assertRaises(ReferenceRefusal):
                ShowdownWorkerFactory('c', 'd', 'e', 's', history_prefix_reuse=bad)
        self.assertTrue(ShowdownWorkerFactory('c', 'd', 'e', 's',
            history_particles=32, history_prefix_reuse=True).history_prefix_reuse)

    def test_accuracy_proposal_flag_is_part_of_bank_identity(self):
        self.assertIn('guide_history_accuracy', OPTIONS)
        for before, after in ((False,True),(True,False),(True,True)):
            old, new = self.setup_pair()
            old.factory.guide_history_chance = new.guide_history_chance = True
            old.factory.guide_history_accuracy = before
            new.guide_history_accuracy = after
            old.receipt['guided_history_accuracy'] = before
            particles = old.particles[:]
            with patch('pokezero.mcts_eval.paper_reference_prefix_frontier.decision_state',
                       side_effect=lambda obs, player: obs):
                next_population = HypotheticalHistoryPopulation(new)
                accepted = next_population.adopt_complete_prefix(old)
            self.assertEqual(accepted, before == after)
            if accepted:
                self.assertEqual(old.particles, [])
                self.assertTrue(next_population.receipt['guided_history_accuracy'])
                next_population.close()
            else:
                self.assertEqual(old.particles, particles)
                self.assertEqual(next_population.receipt['guided_history_accuracy'], after)
                old.close()


if __name__ == '__main__':
    unittest.main()
